"""Durable inbox, operator labels, and asynchronous triage jobs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import logging
from os.path import relpath
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
import threading
import time
import uuid

from decision import DEFAULT_WORKERS, decide, reason_review

RETRY_DELAY_SECONDS = 5
LEVELS = ("top", "urgent", "normal", "non_urgent")
_LEVEL_ORDER = {level: index for index, level in enumerate(LEVELS)}
LOG = logging.getLogger(__name__)


class StoreConflict(ValueError):
    """The submitted request no longer matches durable inbox state."""


def _pending_reason(decision):
    if decision["mode"] == "zoro" and decision.get("review"):
        return {"status": "pending"}
    return None


class Inbox:
    def __init__(self, home, image_root, manifest, workers=DEFAULT_WORKERS):
        self.home = Path(home)
        self.images = self.home / "uploads"
        self.logs = self.home / "logs"
        self.images.mkdir(parents=True, exist_ok=True)
        self.image_root = Path(image_root)
        self.fixtures = json.loads(Path(manifest).read_text())
        self.examples = {r["example"]: r for r in self.fixtures if "example" in r}
        self.db_path = self.home / "inbox.sqlite"
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS cards(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS journal(seq INTEGER PRIMARY KEY, stream TEXT, payload TEXT);
            CREATE INDEX IF NOT EXISTS journal_stream_seq ON journal(stream, seq);
            CREATE TABLE IF NOT EXISTS receipts(
                scope TEXT NOT NULL,
                request_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                result TEXT NOT NULL,
                PRIMARY KEY(scope, request_id)
            );
        """)
        self.changed = threading.Condition(threading.RLock())
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.cards = {k: json.loads(v) for k, v in self.db.execute("SELECT id,data FROM cards")}
        meta = {k: json.loads(v) for k, v in self.db.execute("SELECT key,data FROM meta")}
        self.rules = meta.get("rules", [])
        self.revision = meta.get("revision", 0)
        self.labels = meta.get("labels", 0)
        self.burst = meta.get("burst")
        self.outage_recovery = meta.get("outage_recovery", False)
        self.version = 0
        self.cached = None
        self._restore_burst()
        self._restore_retries()

        self._exported = {}
        self._export_requested = threading.Event()
        self._export_stopping = threading.Event()
        self._export_thread = threading.Thread(target=self._export_loop, name="journal-export", daemon=True)
        self._export_thread.start()
        self._export_requested.set()
        self._queued_retries = set()
        self._stopping = False
        self._retry_thread = threading.Thread(target=self._retry_loop, name="decision-retry", daemon=True)
        self._reason_thread = threading.Thread(target=self._reason_loop, name="reason-review", daemon=True)
        with self.changed:
            for c in self.cards.values():
                if c["status"] == "pending":
                    self.pool.submit(self.work, c["id"], self.revision)
        self._retry_thread.start()
        self._reason_thread.start()

    def close(self):
        with self.changed:
            if self._stopping:
                return
            self._stopping = True
            self.changed.notify_all()
        self._retry_thread.join()
        self._reason_thread.join()
        self.pool.shutdown()
        self._export_stopping.set()
        self._export_requested.set()
        self._export_thread.join()
        self.db.close()

    def _export_loop(self):
        reported_error = False
        while True:
            self._export_requested.wait()
            self._export_requested.clear()
            stopping = self._export_stopping.is_set()
            try:
                self.export()
            except (OSError, sqlite3.Error) as error:
                if not reported_error:
                    LOG.warning("JSONL journal export failed: %s", error)
                    reported_error = True
                self._export_stopping.wait(0.1)
                self._export_requested.set()
            else:
                reported_error = False
            if stopping:
                return

    def export(self):
        # Each stream stays readable through interrupted writes and retries.
        self.logs.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as db:
            streams = db.execute("SELECT stream,MAX(seq) FROM journal GROUP BY stream").fetchall()
            error = None
            for stream, last in streams:
                if last <= self._exported.get(stream, 0):
                    continue
                try:
                    self._export_stream(db, stream, last)
                except OSError as problem:
                    error = error or problem
                else:
                    self._exported[stream] = last
            if error is not None:
                raise error

    def _export_stream(self, db, stream, last):
        temp = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.logs, delete=False) as fp:
                temp = Path(fp.name)
                rows = db.execute(
                    "SELECT payload FROM journal WHERE stream=? AND seq<=? ORDER BY seq", (stream, last)
                )
                for payload, in rows:
                    fp.write(payload + "\n")
            temp.replace(self.logs / (stream + ".jsonl"))
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def _restore_burst(self):
        if not self.burst or "revisions" in self.burst:
            return
        # Recover first completions for an inbox written before they were tracked explicitly.
        batch = {key: c for key, c in self.cards.items() if c["batch"] == self.burst["id"]}
        started_at = self.burst["started_at"]
        started_revision = 0
        first = {}
        for payload, in self.db.execute("SELECT payload FROM journal ORDER BY seq"):
            entry = json.loads(payload)
            if entry["event"] == "rule_added" and entry["ts"] <= started_at:
                started_revision = entry["revision"]
            elif entry["event"] == "decision" and entry["id"] in batch:
                first.setdefault(entry["id"], entry)
        if first and self.burst["done"] != len(first):
            self.burst["elapsed_ms"] = round((max(e["ts"] for e in first.values()) - started_at) * 1000)
        self.burst.update(
            done=len(first),
            fallback=sum(entry["decision"]["mode"] == "fallback" for entry in first.values()),
            started_revision=started_revision,
            revisions=sorted({entry["revision"] for entry in first.values()}),
        )
        with self.db:
            for key, entry in first.items():
                batch[key]["burst_first_revision"] = entry["revision"]
                self.save(batch[key])
            self.meta("burst", self.burst)

    def _restore_retries(self):
        legacy = {key for key, card in self.cards.items() if "recovery" not in card
                  and (card["decision"] or {}).get("mode") == "fallback"}
        completed = {}
        if legacy:
            for payload, in self.db.execute("SELECT payload FROM journal ORDER BY seq DESC"):
                entry = json.loads(payload)
                key = entry.get("id")
                if entry["event"] == "decision" and key in legacy:
                    completed.setdefault(key, entry["ts"])
                    if len(completed) == len(legacy):
                        break
        with self.db:
            for key, card in self.cards.items():
                recovery = card.get("recovery")
                if key in legacy:
                    card["recovery"] = self._new_recovery(card, completed.get(key, time.time()))
                    self.save(card)
                elif recovery and recovery["status"] == "retrying":
                    # The model call may have started before a crash. Consume that attempt.
                    recovery.update(status="failed", finished_at=time.time(), reason="interrupted")
                    self.save(card)
                    self._journal_retry(card, self.rules[:recovery["revision"]], None, recovery)

    def _retry_loop(self):
        with self.changed:
            while not self._stopping:
                now = time.time()
                next_due = None
                for key, card in self.cards.items():
                    due = self._retry_due(card)
                    if due is None or key in self._queued_retries:
                        continue
                    if due <= now:
                        self._queued_retries.add(key)
                        self.pool.submit(self._retry, key)
                    elif next_due is None or due < next_due:
                        next_due = due
                self.changed.wait(None if next_due is None else max(0, next_due - time.time()))

    def _retry_due(self, card):
        recovery = card.get("recovery")
        if card["status"] == "done" and recovery and recovery["status"] == "scheduled":
            return recovery["due_at"]

    def _new_recovery(self, card, completed_at):
        waiting = card["example"] == "outage" and not self.outage_recovery
        return dict(status="waiting_for_outage" if waiting else "scheduled",
                    due_at=completed_at + RETRY_DELAY_SECONDS)

    def _retry(self, key):
        with self.changed:
            self._queued_retries.discard(key)
            card = self.cards[key]
            due = self._retry_due(card)
            now = time.time()
            if self._stopping or due is None or due > now:
                self.changed.notify_all()
                return
            revision = self.revision
            rules = self.rules.copy()
            attempt = dict(card["recovery"], status="retrying", attempted_at=now, revision=revision)
            card = dict(card, recovery=attempt)
            with self.db:
                self.save(card)
            self.cards[key] = card
            self.publish()

        try:
            result = self._decide_card(card, rules, force_fallback=False)
        except Exception:
            result = None

        with self.changed:
            card = self.cards[key]
            owns_attempt = card.get("recovery") == attempt
            if not owns_attempt or revision != self.revision:
                status = "superseded"
            elif result and result["mode"] == "zoro":
                status = "recovered"
                card = dict(card, decision=result, urgency=result["urgency"], reason_review=_pending_reason(result))
            else:
                status = "failed"
            recovery = dict(attempt, status=status, finished_at=time.time())
            if result is None:
                recovery["reason"] = "error"
            with self.db:
                if owns_attempt:
                    card = dict(card, recovery=recovery)
                    self.save(card)
                self._journal_retry(card, rules, result, recovery)
            if owns_attempt:
                self.cards[key] = card
            self.publish(journal=True)

    def _next_reason_card(self):
        if any(card["status"] == "pending" for card in self.cards.values()):
            return None
        # A running item found here was interrupted by a shutdown and can be resumed.
        candidates = (
            card for card in self.cards.values()
            if (card.get("reason_review") or {}).get("status") in ("pending", "running")
        )
        return min(candidates, default=None, key=lambda card: (
            _LEVEL_ORDER.get(card["human_urgency"] or card["urgency"], len(LEVELS)), card["created_at"],
        ))

    def _reason_loop(self):
        while True:
            with self.changed:
                card = self._next_reason_card()
                while card is None and not self._stopping:
                    self.changed.wait()
                    card = self._next_reason_card()
                if self._stopping:
                    return
                key = card["id"]
                revision = card["rule_revision"]
                attempt = {"status": "running"}
                card = dict(card, reason_review=attempt)
                with self.db:
                    self.save(card)
                self.cards[key] = card
                self.publish()

            try:
                result = reason_review(card["text"], Path(card["path"]).read_bytes(), card["decision"])
            except Exception as error:
                LOG.warning("Reason review failed for %s: %s", key, type(error).__name__)
                result = None

            with self.changed:
                card = self.cards[key]
                if revision != self.revision or card.get("reason_review") is not attempt:
                    continue
                review = {"status": "unavailable"} if result is None else dict(result, status="done")
                card = dict(card, reason_review=review)
                with self.db:
                    self.save(card)
                    self.journal("reason-reviews", event="reason_review", id=key, revision=revision, review=review)
                self.cards[key] = card
                self.publish(journal=True)

    def journal(self, stream, **data):
        payload = json.dumps(dict(ts=time.time(), **data), ensure_ascii=False)
        self.db.execute("INSERT INTO journal(stream,payload) VALUES(?,?)", (stream, payload))

    def save(self, card):
        self.db.execute("INSERT OR REPLACE INTO cards VALUES(?,?)", (card["id"], json.dumps(card,ensure_ascii=False)))

    def meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))

    def publish(self, journal=False):
        self.cached = None
        self.version += 1
        self.changed.notify_all()
        if journal:
            self._export_requested.set()

    def state(self):
        with self.changed:
            if self.cached is None:
                def rank(c):
                    if c["status"] == "pending":
                        return (0, 0, False, c["created_at"])
                    level = c["human_urgency"] or c["urgency"]
                    review = (c.get("reason_review") or {}).get("needs_review", (c["decision"] or {}).get("review", False))
                    needs_review = c["human_urgency"] is None and review
                    return (1, _LEVEL_ORDER.get(level, len(LEVELS)), not needs_review, c["created_at"])
                private = ("path", "example", "batch", "burst_first_revision")
                cards = [{k:v for k,v in c.items() if k not in private} for c in sorted(self.cards.values(),key=rank)]
                self.cached = json.dumps(dict(cards=cards,rules=self.rules,revision=self.revision,labels=self.labels,
                                              outage_recovery=self.outage_recovery,burst=self.burst,version=self.version),
                                         ensure_ascii=False,separators=(",",":")).encode()
            return self.cached

    def new(self, text, path, source, scenario_id=None, example=None, batch=None, key=None):
        key = key or uuid.uuid4().hex
        return dict(id=key,text=text,path=str(path),source=source,scenario_id=scenario_id,example=example,
                    batch=batch,image_url=f"/api/reports/{key}/image",created_at=time.time(),status="pending",
                    urgency=None,human_urgency=None,decision=None,recovery=None,rule_revision=self.revision)

    def _journal_input(self, card):
        if card["source"] == "phone":
            root, root_name = self.home, "DEMO_HOME"
        else:
            root, root_name = self.image_root, "CRISIS_ROOT"
        return dict(text=card["text"], image=Path(relpath(card["path"], root)).as_posix(),
                    image_root=root_name, scenario=card["scenario_id"])

    def _decision_stream(self, card):
        return "scenario-" + card["example"] if card["source"] == "example" else "decisions"

    def _journal_retry(self, card, rules, decision, recovery):
        self.journal(self._decision_stream(card), event="decision_retry", id=card["id"],
                     input=self._journal_input(card), revision=recovery["revision"], rules=rules,
                     decision=decision, human=card["human_urgency"], recovery=recovery)

    def _receipt(self, scope, request_id, fingerprint):
        if request_id is None:
            return None
        row = self.db.execute(
            "SELECT fingerprint,result FROM receipts WHERE scope=? AND request_id=?", (scope, request_id)
        ).fetchone()
        if row is None:
            return None
        if row[0] != fingerprint:
            raise StoreConflict("같은 요청 ID에 다른 입력을 사용할 수 없습니다.")
        return json.loads(row[1])

    def _record_receipt(self, scope, request_id, fingerprint, result):
        if request_id is not None:
            self.db.execute(
                "INSERT INTO receipts VALUES(?,?,?,?)", (scope, request_id, fingerprint, json.dumps(result))
            )

    def upload(self, text, image, request_id=None, *, extension=".jpg"):
        if extension not in (".jpg", ".png", ".webp"):
            raise ValueError("사진 형식을 확인하세요.")
        fingerprint = None
        if request_id is not None:
            content = hashlib.sha256(text.encode("utf-8"))
            content.update(hashlib.sha256(image).digest())
            fingerprint = content.hexdigest()
            with self.changed:
                previous = self._receipt("upload", request_id, fingerprint)
                if previous is not None:
                    return previous

        key = uuid.uuid4().hex
        path = self.images / (key + extension)
        accepted = False
        try:
            path.write_bytes(image)
            with self.changed:
                previous = self._receipt("upload", request_id, fingerprint)
                if previous is not None:
                    return previous
                card = self.new(text, path, "phone", key=key)
                with self.db:
                    self.save(card)
                    self._record_receipt("upload", request_id, fingerprint, key)
                accepted = True
                self.cards[key] = card
                self.pool.submit(self.work, key, self.revision)
                self.publish()
            return key
        finally:
            if not accepted:
                path.unlink(missing_ok=True)

    def add_examples(self, kind=None, request_id=None):
        fingerprint = json.dumps(kind)
        with self.changed:
            previous = self._receipt("examples", request_id, fingerprint)
            if previous is not None:
                return previous
            if kind is None and self.burst and self.burst["done"] < self.burst["total"]:
                raise ValueError("이미 폭주 시뮬레이션을 처리하고 있습니다.")
            rows = self.fixtures if kind is None else [self.examples[kind]]
            batch = uuid.uuid4().hex if kind is None else None
            paths = [self.image_root / row["image_path"] for row in rows]
            if any(not path.is_file() for path in paths):
                raise ValueError("시뮬레이션 사진 경로를 확인하세요.")
            cards = [self.new(r["text"],p,"burst" if batch else "example",r["id"],kind,batch) for r,p in zip(rows,paths)]
            ids = [card["id"] for card in cards]
            burst = dict(id=batch, total=len(ids), done=0, elapsed_ms=0, started_at=time.time(),
                         started_revision=self.revision, revisions=[], fallback=0) if batch else None
            with self.db:
                for card in cards:
                    self.save(card)
                if burst is not None:
                    self.meta("burst", burst)
                self._record_receipt("examples", request_id, fingerprint, ids)
            self.cards.update(zip(ids, cards))
            if burst is not None:
                self.burst = burst
            for key in ids:
                self.pool.submit(self.work, key, self.revision)
            self.publish()
            return ids

    def work(self, key, revision):
        with self.changed:
            if revision != self.revision or self.cards[key]["status"] != "pending":
                return
            c = self.cards[key].copy()
            rules = self.rules.copy()
            force_fallback = c["example"] == "outage" and (c["decision"] is None or not self.outage_recovery)
        try:
            result = self._decide_card(c, rules, force_fallback=force_fallback)
        except Exception:
            result = decide(c["text"], b"", rules, force_fallback=True)
            result["reasons"].append("입력 또는 판정 처리 오류")
        with self.changed:
            if revision != self.revision or self.cards[key]["status"] != "pending":
                return
            c = self.cards[key].copy()
            now = time.time()
            if c["decision"] is None and self.burst:
                # The latest burst necessarily started before this completion. If its end
                # follows receipt, the two intervals overlap, including for phone uploads.
                end = self.burst.get("finished_at")
                if end is None:
                    end = now if self.burst["done"] < self.burst["total"] else (
                        self.burst["started_at"] + self.burst["elapsed_ms"] / 1000
                    )
                if c["batch"] or c["created_at"] <= end:
                    c["first_decision_with_wait_ms"] = round((now - c["created_at"]) * 1000, 1)
            c.update(decision=result,status="done",urgency=result["urgency"],rule_revision=revision)
            c["recovery"] = self._new_recovery(c, now) if result["mode"] == "fallback" else None
            c["reason_review"] = _pending_reason(result)
            burst = None
            if self.burst and c["batch"] == self.burst["id"] and "burst_first_revision" not in c:
                c["burst_first_revision"] = revision
                burst = self.burst | {
                    "done": self.burst["done"] + 1,
                    "elapsed_ms": round((now - self.burst["started_at"]) * 1000),
                    "fallback": self.burst["fallback"] + int(result["mode"] == "fallback"),
                    "revisions": sorted(set(self.burst["revisions"]) | {revision}),
                }
                if burst["done"] == burst["total"]:
                    burst["finished_at"] = now
            with self.db:
                self.save(c)
                if burst is not None:
                    self.meta("burst", burst)
                self.journal(self._decision_stream(c),event="decision",id=key,input=self._journal_input(c),
                             revision=revision,rules=rules,decision=result,human=c["human_urgency"])
            self.cards[key] = c
            if burst is not None:
                self.burst = burst
            self.publish(journal=True)

    def _decide_card(self, card, rules, *, force_fallback):
        return decide(card["text"], Path(card["path"]).read_bytes(), rules, force_fallback=force_fallback)

    def set_outage_recovery(self, enabled: bool):
        with self.changed:
            if self.outage_recovery == enabled:
                return
            status = "scheduled" if enabled else "waiting_for_outage"
            cards = {}
            for key, card in self.cards.items():
                recovery = card.get("recovery")
                if card["example"] == "outage" and recovery and recovery["status"] in (
                    "scheduled", "waiting_for_outage"
                ):
                    cards[key] = dict(card, recovery=dict(recovery, status=status))
            with self.db:
                self.meta("outage_recovery", enabled)
                for card in cards.values():
                    self.save(card)
            self.outage_recovery = enabled
            self.cards.update(cards)
            self.publish()

    def rule(self,text):
        with self.changed:
            if text in self.rules or len(self.rules)>=8:
                raise ValueError("중복 조건이거나 최대 8개 조건에 도달했습니다.")
            rules = [*self.rules, text]
            revision = self.revision + 1
            cards = {key: dict(c, status="pending", reason_review=None) for key, c in self.cards.items()}
            with self.db:
                self.meta("rules", rules)
                self.meta("revision", revision)
                for card in cards.values():
                    self.save(card)
                self.journal("rules", event="rule_added", revision=revision, text=text)
            self.rules = rules
            self.revision = revision
            self.cards = cards
            for key in self.cards:
                self.pool.submit(self.work, key, revision)
            self.publish(journal=True)

    def label(self, key, level, expected_revision=None):
        with self.changed:
            c = self.cards[key]
            if expected_revision is not None and (type(expected_revision) is not int or expected_revision != self.revision):
                raise StoreConflict("규칙이 변경됐습니다. 최신 판정을 확인하세요.")
            if c["status"] != "done":
                raise ValueError("판정 후 수정할 수 있습니다.")
            if c["human_urgency"] == level:
                return
            count = self.labels + int(c["human_urgency"] is None)
            c = dict(c, human_urgency=level)
            with self.db:
                if count != self.labels:
                    self.meta("labels", count)
                self.save(c)
                self.journal("labels",event="label",id=key,input=self._journal_input(c),
                             revision=c["rule_revision"],rules=self.rules,decision=c["decision"],human=level,
                             reason_review=c.get("reason_review"))
            self.cards[key] = c
            self.labels = count
            self.publish(journal=True)
