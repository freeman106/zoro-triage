from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from unittest.mock import patch

import store


def outcome(mode="zoro"):
    return {"urgency": "normal", "mode": mode, "review": False, "reasons": []}


class StoreTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.home = root / "inbox"
        self.manifest = root / "fixtures.json"
        self.image_root = root / "images"
        self.image_root.mkdir()
        (self.image_root / "test.jpg").write_bytes(b"fixture")
        self.manifest.write_text(json.dumps([
            {"id": "one", "text": "first", "image_path": "test.jpg", "example": "top"},
            {"id": "two", "text": "second", "image_path": "test.jpg", "example": "mismatch"},
        ]))
        model = patch.object(store, "decide", side_effect=lambda *args, **kwargs: outcome())
        self.model = model.start()
        self.addCleanup(model.stop)

    def open_inbox(self):
        inbox = store.Inbox(self.home, self.image_root, self.manifest, workers=2)

        def close():
            if hasattr(inbox, "close"):
                inbox.close()
            else:
                inbox.pool.shutdown()
                inbox.db.close()

        self.addCleanup(close)
        return inbox

    def wait_for(self, read, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            value = read()
            if predicate(value):
                return value
            time.sleep(0.01)
        self.fail(f"Condition never reached: {read()!r}")

    def settled(self, inbox, count, revision):
        return self.wait_for(
            lambda: json.loads(inbox.state()),
            lambda state: len(state["cards"]) == count and all(
                card["status"] == "done" and card["rule_revision"] == revision
                for card in state["cards"]
            ),
        )

    def log(self, stream):
        path = self.home / "logs" / (stream + ".jsonl")
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()]

    def card(self, inbox, key):
        return next(card for card in json.loads(inbox.state())["cards"] if card["id"] == key)

    def test_reason_waits_for_initial_batch_and_processes_top_first(self):
        allow_top = Event()
        reason_started = Event()
        allow_reason = Event()
        calls = []
        clear = {"topic": "clear", "probability": 0.95, "needs_review": False}

        def model(text, image, rules, *, force_fallback):
            if text == "second":
                allow_top.wait(2)
                return outcome() | {"urgency": "top", "review": True}
            return outcome() | {"review": True}

        def reason(text, image, decision):
            calls.append(text)
            reason_started.set()
            allow_reason.wait(2)
            return clear

        self.model.side_effect = model
        with patch.object(store, "reason_review", side_effect=reason):
            inbox = self.open_inbox()
            first, top = inbox.add_examples()
            try:
                first_pass = self.wait_for(lambda: self.card(inbox, first), lambda card: card["status"] == "done")
                self.assertEqual(first_pass["reason_review"]["status"], "pending")
                self.assertFalse(reason_started.is_set())
                allow_top.set()
                self.assertTrue(reason_started.wait(1))
                state = self.settled(inbox, 2, 0)
                self.assertEqual([c["id"] for c in state["cards"]], [top, first])
                self.assertEqual(state["cards"][0]["reason_review"]["status"], "running")
                self.assertEqual(state["burst"]["done"], 2)
                inbox.label(top, "urgent")
            finally:
                allow_top.set()
                allow_reason.set()
            logs = self.wait_for(lambda: self.log("reason-reviews"), lambda entries: len(entries) == 2)
            self.assertEqual(calls, ["second", "first"])
            self.assertEqual([entry["id"] for entry in logs], [top, first])
            self.assertEqual(self.card(inbox, top)["urgency"], "top")
            self.assertEqual(self.card(inbox, top)["human_urgency"], "urgent")
            self.assertEqual(self.card(inbox, first)["reason_review"], clear | {"status": "done"})
            self.assertEqual(json.loads(inbox.state())["burst"], state["burst"])
            inbox.close()

    def test_rule_change_discards_old_reason_and_keeps_failed_review_flag(self):
        reason_started = Event()
        allow_old_reason = Event()
        attempted = []

        def model(text, image, rules, *, force_fallback):
            return outcome() | {"review": True, "rules": rules}

        def reason(text, image, decision):
            attempted.append(decision["rules"])
            if not decision["rules"]:
                reason_started.set()
                allow_old_reason.wait(2)
                return {"topic": "clear", "probability": 0.95, "needs_review": False}
            return None

        self.model.side_effect = model
        with patch.object(store, "reason_review", side_effect=reason):
            inbox = self.open_inbox()
            key = inbox.add_examples("top")[0]
            try:
                self.assertTrue(reason_started.wait(1))
                inbox.rule("새로운 조건")
                self.settled(inbox, 1, 1)
                self.assertEqual(self.card(inbox, key)["reason_review"]["status"], "pending")
            finally:
                allow_old_reason.set()
            card = self.wait_for(
                lambda: self.card(inbox, key),
                lambda card: card["reason_review"]["status"] == "unavailable",
            )
            self.assertTrue(card["decision"]["review"])
            self.assertEqual(attempted, [[], ["새로운 조건"]])
            entries = self.wait_for(lambda: self.log("reason-reviews"), bool)
            self.assertEqual([entry["revision"] for entry in entries], [1])
            inbox.close()

    def test_reason_resumes_after_restart_and_orders_effective_review(self):
        self.model.side_effect = [outcome(), outcome() | {"review": True}]
        with patch.object(store.Inbox, "_next_reason_card", return_value=None):
            inbox = self.open_inbox()
            older = inbox.add_examples("top")[0]
            self.settled(inbox, 1, 0)
            needs_review = inbox.add_examples("mismatch")[0]
            state = self.settled(inbox, 2, 0)
            self.assertEqual([c["id"] for c in state["cards"]], [needs_review, older])
            with inbox.changed, inbox.db:
                interrupted = dict(inbox.cards[needs_review], reason_review={"status": "running"})
                inbox.save(interrupted)
            inbox.close()

        result = {"topic": "clear", "probability": 0.9, "needs_review": False}
        with patch.object(store, "reason_review", return_value=result) as reason:
            reopened = self.open_inbox()
            state = self.wait_for(
                lambda: json.loads(reopened.state()),
                lambda state: state["cards"][-1]["reason_review"] == result | {"status": "done"},
            )
            self.assertEqual([c["id"] for c in state["cards"]], [older, needs_review])
            self.assertEqual(reason.call_count, 1)
            reopened.close()

    def test_retry_recovers_in_background_and_preserves_manual_label_and_burst(self):
        retry_started = Event()
        release = Event()
        attempts = []

        def model(text, image, rules, *, force_fallback):
            if text == "second":
                return outcome()
            attempts.append((image, rules, force_fallback))
            if len(attempts) == 1:
                return outcome("fallback")
            retry_started.set()
            release.wait(2)
            return outcome() | {"urgency": "urgent"}

        self.model.side_effect = model
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.1, create=True):
            inbox = self.open_inbox()
            key, _ = inbox.add_examples()
            burst = self.settled(inbox, 2, 0)["burst"]
            self.assertEqual(burst["fallback"], 1)
            try:
                self.assertTrue(retry_started.wait(1))
                card = self.card(inbox, key)
                self.assertEqual(card["status"], "done")
                self.assertEqual(card["decision"]["mode"], "fallback")
                self.assertEqual(card["recovery"]["status"], "retrying")
                inbox.label(key, "top")
            finally:
                release.set()
            recovered = self.wait_for(lambda: self.card(inbox, key), lambda card: card["urgency"] == "urgent")
            self.assertEqual(recovered["recovery"]["status"], "recovered")
            self.assertGreaterEqual(recovered["recovery"]["attempted_at"], recovered["recovery"]["due_at"])
            self.assertEqual(recovered["human_urgency"], "top")
            self.assertEqual(recovered["decision"]["mode"], "zoro")
            self.assertEqual(json.loads(inbox.state())["burst"], burst)
            self.assertEqual(attempts, [(b"fixture", [], False)] * 2)
            logs = self.wait_for(lambda: self.log("decisions"), lambda entries: len(entries) == 3)
            matching = [entry for entry in logs if entry["id"] == key]
            self.assertEqual([entry["event"] for entry in matching], ["decision", "decision_retry"])
            self.assertEqual([entry["decision"]["mode"] for entry in matching], ["fallback", "zoro"])
            self.assertEqual(matching[1]["human"], "top")

    def test_failed_retry_keeps_original_decision_and_is_not_repeated_after_restart(self):
        original = outcome("fallback") | {"reasons": ["initial reason"]}
        later = outcome("fallback") | {"reasons": ["retry reason"], "urgency": "urgent"}
        self.model.side_effect = [original, later]
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.03, create=True):
            inbox = self.open_inbox()
            key = inbox.upload("report", b"photo")
            failed = self.wait_for(
                lambda: self.card(inbox, key), lambda card: (card.get("recovery") or {}).get("status") == "failed"
            )
            self.assertEqual(failed["decision"], original)
            self.assertEqual(failed["urgency"], "normal")
            inbox.close()
            reopened = self.open_inbox()
            self.assertEqual(self.card(reopened, key), failed)
            time.sleep(0.06)
            self.assertEqual(self.model.call_count, 2)
            logs = self.wait_for(lambda: self.log("decisions"), lambda entries: len(entries) == 2)
            retries = [entry for entry in logs if entry["event"] == "decision_retry"]
            self.assertEqual([entry["decision"] for entry in retries], [later])

    def test_fresh_rule_fallback_retries_once_after_prior_success_or_failure(self):
        fallback = outcome("fallback")
        self.model.side_effect = [fallback, outcome(), fallback, fallback, fallback, outcome()]
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.03):
            inbox = self.open_inbox()
            key = inbox.upload("report", b"photo")
            for revision, status in enumerate(("recovered", "failed", "recovered")):
                if revision:
                    inbox.rule(f"rule {revision}")
                card = self.wait_for(
                    lambda: self.card(inbox, key),
                    lambda card: card["rule_revision"] == revision
                    and (card.get("recovery") or {}).get("status") == status,
                )
                self.assertEqual(card["decision"]["mode"], "zoro" if status == "recovered" else "fallback")
                self.assertEqual(card["recovery"]["revision"], revision)
            logs = self.wait_for(lambda: self.log("decisions"), lambda entries: len(entries) == 6)
            retries = [entry["revision"] for entry in logs if entry["event"] == "decision_retry"]
            self.assertEqual(retries, [0, 1, 2])
            self.assertEqual(self.model.call_count, 6)

    def test_simulated_outage_recovery_can_be_armed_after_deadline_and_survives_restart(self):
        rows = json.loads(self.manifest.read_text())
        rows.append(dict(rows[0], id="outage", example="outage"))
        self.manifest.write_text(json.dumps(rows))
        self.model.side_effect = lambda *args, force_fallback: outcome("fallback" if force_fallback else "zoro")
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.05, create=True):
            inbox = self.open_inbox()
            key, = inbox.add_examples("outage")
            snapshot = self.settled(inbox, 1, 0)
            self.assertFalse(snapshot["outage_recovery"])
            recovery = snapshot["cards"][0]["recovery"]
            self.assertEqual(recovery["status"], "waiting_for_outage")
            self.wait_for(time.time, lambda now: now > recovery["due_at"])
            self.assertEqual(self.model.call_count, 1)
            inbox.close()

            reopened = self.open_inbox()
            reopened.set_outage_recovery(True)
            recovered = self.wait_for(lambda: self.card(reopened, key), lambda card: card["decision"]["mode"] == "zoro")
            self.assertEqual(recovered["recovery"]["due_at"], recovery["due_at"])
            reopened.close()

            restarted = self.open_inbox()
            self.assertTrue(json.loads(restarted.state())["outage_recovery"])
            second, = restarted.add_examples("outage")
            self.wait_for(
                lambda: self.card(restarted, second),
                lambda card: (card.get("recovery") or {}).get("status") == "recovered",
            )
            self.assertEqual([call.kwargs["force_fallback"] for call in self.model.call_args_list],
                             [True, False, True, False])

    def test_scheduled_retry_resumes_after_restart(self):
        self.model.side_effect = [outcome("fallback"), outcome()]
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.2, create=True):
            inbox = self.open_inbox()
            key = inbox.upload("report", b"photo")
            before = self.settled(inbox, 1, 0)["cards"][0]["recovery"]
            self.assertEqual(before["status"], "scheduled")
            inbox.close()
            self.assertEqual(self.model.call_count, 1)

            reopened = self.open_inbox()
            after = self.wait_for(
                lambda: self.card(reopened, key), lambda card: card["recovery"]["status"] == "recovered"
            )
            self.assertEqual(after["recovery"]["due_at"], before["due_at"])
            self.assertEqual(self.model.call_count, 2)

    def test_restart_marks_interrupted_attempt_failed_without_repeating_it(self):
        self.model.side_effect = [outcome("fallback")]
        inbox = self.open_inbox()
        key = inbox.upload("report", b"photo")
        saved = self.settled(inbox, 1, 0)["cards"][0]
        inbox.close()

        with closing(sqlite3.connect(self.home / "inbox.sqlite")) as db, db:
            payload, = db.execute("SELECT data FROM cards WHERE id=?", (key,)).fetchone()
            card = json.loads(payload)
            card["recovery"].update(status="retrying", attempted_at=time.time(), revision=0)
            db.execute("UPDATE cards SET data=? WHERE id=?", (json.dumps(card), key))

        reopened = self.open_inbox()
        after = self.card(reopened, key)
        self.assertEqual(after["decision"], saved["decision"])
        self.assertEqual(after["recovery"]["status"], "failed")
        self.assertEqual(after["recovery"]["reason"], "interrupted")
        logs = self.wait_for(lambda: self.log("decisions"), lambda entries: len(entries) == 2)
        self.assertEqual(logs[1]["event"], "decision_retry")
        self.assertEqual(logs[1]["recovery"], after["recovery"])
        self.assertEqual(self.model.call_count, 1)

    def test_retry_does_not_overwrite_new_rule_revision(self):
        retry_started = Event()
        release = Event()

        def model(text, image, rules, *, force_fallback):
            if rules:
                mode = "fallback" if self.model.call_count == 3 else "zoro"
                return outcome(mode) | {"urgency": "non_urgent"}
            if self.model.call_count == 1:
                return outcome("fallback")
            retry_started.set()
            release.wait(2)
            return outcome() | {"urgency": "urgent"}

        self.model.side_effect = model
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.03, create=True):
            inbox = self.open_inbox()
            key = inbox.upload("report", b"photo")
            try:
                self.assertTrue(retry_started.wait(1))
                inbox.rule("new rule")
                newer = self.wait_for(
                    lambda: self.card(inbox, key),
                    lambda card: card["rule_revision"] == 1 and card["recovery"]["status"] == "recovered",
                )
                self.assertEqual(newer["urgency"], "non_urgent")
            finally:
                release.set()
            logs = self.wait_for(lambda: self.log("decisions"), lambda entries: len(entries) == 4)
            self.assertEqual(logs[-1]["revision"], 0)
            self.assertEqual(logs[-1]["recovery"]["status"], "superseded")
            self.assertEqual(self.card(inbox, key), newer)

    def test_retry_keeps_replacement_recovery_from_new_rule_revision(self):
        retry_started = Event()
        release = Event()

        def model(text, image, rules, *, force_fallback):
            if rules:
                return outcome() | {"urgency": "non_urgent"}
            if self.model.call_count == 1:
                return outcome("fallback")
            retry_started.set()
            release.wait(2)
            return outcome() | {"urgency": "urgent"}

        self.model.side_effect = model
        with patch.object(store, "RETRY_DELAY_SECONDS", 0.03):
            inbox = self.open_inbox()
            key = inbox.upload("report", b"photo")
            try:
                self.assertTrue(retry_started.wait(1))
                inbox.rule("new rule")
                newer = self.settled(inbox, 1, 1)["cards"][0]
                self.assertIsNone(newer["recovery"])
            finally:
                release.set()

            logs = self.wait_for(lambda: self.log("decisions"), lambda entries: len(entries) == 3)
            self.assertEqual(logs[-1]["event"], "decision_retry")
            self.assertEqual(logs[-1]["recovery"]["status"], "superseded")
            self.assertEqual(logs[-1]["revision"], 0)
            self.assertEqual(newer["urgency"], "non_urgent")
            self.assertEqual(self.card(inbox, key), newer)

    def test_pending_and_done_cards_follow_urgency_review_and_fifo(self):
        decisions = {
            "older checked": ("urgent", False),
            "older review": ("urgent", True),
            "lower review": ("normal", True),
            "newer review": ("urgent", True),
            "top checked": ("top", False),
            "latest review": ("urgent", True),
        }
        release = Event()

        def model(text, *args, **kwargs):
            if text.startswith("pending"):
                release.wait(2)
                return outcome()
            urgency, review = decisions[text]
            return outcome() | {"urgency": urgency, "review": review}

        self.model.side_effect = model
        inbox = self.open_inbox()
        ids = {text: inbox.upload(text, b"photo") for text in decisions}
        self.settled(inbox, len(ids), 0)
        inbox.label(ids["older review"], "urgent")
        inbox.label(ids["lower review"], "top")

        try:
            inbox.upload("pending first", b"photo")
            inbox.upload("pending second", b"photo")
            cards = json.loads(inbox.state())["cards"]
            self.assertEqual([card["text"] for card in cards], [
                "pending first", "pending second", "lower review", "top checked",
                "newer review", "latest review", "older checked", "older review",
            ])
        finally:
            release.set()

    def test_decision_and_label_logs_use_relative_image_references(self):
        inbox = self.open_inbox()
        phone = inbox.upload("new report", b"phone photo")
        example, = inbox.add_examples("top")
        burst = inbox.add_examples()
        self.settled(inbox, 4, 0)
        inbox.label(phone, "urgent")
        inbox.label(example, "top")
        inbox.close()

        roots = {"DEMO_HOME": self.home, "CRISIS_ROOT": self.image_root}
        expected = {
            phone: ("DEMO_HOME", f"uploads/{phone}.jpg", b"phone photo"),
            **{key: ("CRISIS_ROOT", "test.jpg", b"fixture") for key in [example, *burst]},
        }
        entries = {stream: self.log(stream) for stream in ("decisions", "scenario-top", "labels")}
        self.assertCountEqual([entry["id"] for entry in entries["decisions"]], [phone, *burst])
        self.assertEqual([entry["id"] for entry in entries["scenario-top"]], [example])
        self.assertEqual([entry["id"] for entry in entries["labels"]], [phone, example])
        for stream in entries.values():
            for entry in stream:
                with self.subTest(event=entry["event"], id=entry["id"]):
                    root, image, data = expected[entry["id"]]
                    self.assertEqual(entry["input"]["image"], image)
                    self.assertEqual(entry["input"]["image_root"], root)
                    self.assertEqual((roots[root] / image).read_bytes(), data)

    def test_log_projection_retries_without_stranding_committed_work(self):
        inbox = self.open_inbox()
        failed = Event()
        recover = Event()
        original_export = store.Inbox.export

        def flaky_export(instance, *args):
            if not recover.is_set():
                failed.set()
                raise OSError("simulated projection failure")
            return original_export(instance, *args)

        with patch.object(store.Inbox, "export", flaky_export):
            try:
                first = inbox.upload("first report", b"photo")
                self.assertTrue(failed.wait(1))
                second = inbox.upload("second report", b"photo")
                self.settled(inbox, 2, 0)
                inbox.label(first, "urgent")
                self.assertEqual(json.loads(inbox.state())["labels"], 1)
                self.assertEqual(self.log("decisions"), [])
            finally:
                recover.set()

            decisions = self.wait_for(lambda: self.log("decisions"), lambda logs: len(logs) == 2)
            labels = self.wait_for(lambda: self.log("labels"), lambda logs: len(logs) == 1)
        self.assertCountEqual([entry["id"] for entry in decisions], [first, second])
        self.assertEqual(labels[0]["id"], first)

    def test_restart_rebuilds_projection_without_duplicate_entries(self):
        inbox = self.open_inbox()
        key = inbox.upload("first report", b"photo")
        self.settled(inbox, 1, 0)
        self.wait_for(lambda: self.log("decisions"), bool)
        inbox.close()

        path = self.home / "logs" / "decisions.jsonl"
        path.write_text(path.read_text() * 2)
        reopened = self.open_inbox()
        decisions = self.wait_for(lambda: self.log("decisions"), lambda logs: len(logs) == 1)
        self.assertEqual(decisions[0]["id"], key)
        self.settled(reopened, 1, 0)

    def test_projection_failure_keeps_old_file_and_exports_other_streams(self):
        inbox = self.open_inbox()
        first = inbox.upload("first report", b"photo")
        self.settled(inbox, 1, 0)
        original = self.wait_for(lambda: self.log("decisions"), bool)
        failed = Event()
        recover = Event()
        replace = Path.replace

        def flaky_replace(path, target):
            if target.name == "decisions.jsonl" and not recover.is_set():
                failed.set()
                raise OSError("simulated atomic replacement failure")
            return replace(path, target)

        with patch.object(Path, "replace", flaky_replace):
            try:
                second = inbox.upload("second report", b"photo")
                self.settled(inbox, 2, 0)
                inbox.label(first, "urgent")
                self.assertTrue(failed.wait(1))
                self.assertEqual(self.wait_for(lambda: self.log("labels"), bool)[0]["id"], first)
                self.assertEqual(self.log("decisions"), original)
            finally:
                recover.set()
            decisions = self.wait_for(lambda: self.log("decisions"), lambda logs: len(logs) == 2)
        self.assertCountEqual([entry["id"] for entry in decisions], [first, second])

    def test_request_id_replays_concurrent_upload_and_survives_restart(self):
        inbox = self.open_inbox()
        with ThreadPoolExecutor(max_workers=2) as requests:
            ids = list(requests.map(lambda _: inbox.upload("same report", b"photo", request_id="request"), range(2)))
        self.assertEqual(ids[0], ids[1])
        self.settled(inbox, 1, 0)
        self.assertEqual(len(list(inbox.images.iterdir())), 1)
        with self.assertRaises(ValueError):
            inbox.upload("different report", b"photo", request_id="request")
        with self.assertRaises(ValueError):
            inbox.upload("same report", b"different photo", request_id="request")
        example = inbox.add_examples("top", request_id="request")
        self.assertNotIn(ids[0], example)
        self.settled(inbox, 2, 0)
        inbox.close()

        reopened = self.open_inbox()
        self.assertEqual(reopened.upload("same report", b"photo", request_id="request"), ids[0])
        self.assertEqual(reopened.add_examples("top", request_id="request"), example)
        self.settled(reopened, 2, 0)
        self.assertNotEqual(reopened.upload("same report", b"photo"), ids[0])
        self.settled(reopened, 3, 0)

    def test_upload_keeps_verified_extension_in_storage_and_journal(self):
        inbox = self.open_inbox()
        with self.assertRaises(ValueError):
            inbox.upload("report", b"photo", extension="/anything")
        key = inbox.upload("report", b"photo", extension=".webp")
        self.settled(inbox, 1, 0)
        logs = self.wait_for(lambda: self.log("decisions"), bool)
        relative = f"uploads/{key}.webp"
        self.assertEqual(logs[0]["input"]["image"], relative)
        self.assertEqual((self.home / relative).read_bytes(), b"photo")

    def test_burst_card_and_concurrent_phone_keep_their_first_queue_inclusive_latency(self):
        entered = Event()
        release = Event()

        def model(*args, **kwargs):
            entered.set()
            release.wait(2)
            return outcome() | {"ms": 1.2}

        self.model.side_effect = model
        inbox = self.open_inbox()
        try:
            burst_ids = inbox.add_examples()
            self.assertTrue(entered.wait(1))
            phone_id = inbox.upload("during burst", b"photo")
            time.sleep(0.03)
        finally:
            release.set()
        completed = self.settled(inbox, 3, 0)
        first = {c["id"]: c["first_decision_with_wait_ms"] for c in completed["cards"]}
        self.assertEqual(set(first), {*burst_ids, phone_id})
        self.assertGreater(first[phone_id], 20)
        self.assertEqual(self.card(inbox, phone_id)["decision"]["ms"], 1.2)

        late_id = inbox.upload("after burst", b"photo")
        self.settled(inbox, 4, 0)
        inbox.rule("new rule")
        revised = self.settled(inbox, 4, 1)
        for card in revised["cards"]:
            if card["id"] == late_id:
                self.assertNotIn("first_decision_with_wait_ms", card)
            else:
                self.assertEqual(card["first_decision_with_wait_ms"], first[card["id"]])

    def test_burst_receipt_replays_during_and_after_processing(self):
        processing = Event()
        release = Event()

        def slow_model(*args, **kwargs):
            processing.set()
            release.wait(2)
            return outcome()

        self.model.side_effect = slow_model
        inbox = self.open_inbox()
        try:
            ids = inbox.add_examples(request_id="batch")
            self.assertTrue(processing.wait(1))
            self.assertEqual(inbox.add_examples(request_id="batch"), ids)
            with self.assertRaises(ValueError):
                inbox.add_examples("top", request_id="batch")
            with self.assertRaises(ValueError):
                inbox.add_examples(request_id="another-batch")
        finally:
            release.set()
        self.settled(inbox, 2, 0)
        self.assertEqual(inbox.add_examples(request_id="batch"), ids)
        self.settled(inbox, 2, 0)

    def test_stale_label_revision_does_not_change_card_or_count(self):
        inbox = self.open_inbox()
        key = inbox.upload("report", b"photo")
        self.settled(inbox, 1, 0)
        inbox.rule("조건")
        self.settled(inbox, 1, 1)
        with self.assertRaises(ValueError):
            inbox.label(key, "urgent", expected_revision=0)
        snapshot = json.loads(inbox.state())
        self.assertEqual(snapshot["labels"], 0)
        self.assertIsNone(snapshot["cards"][0]["human_urgency"])
        inbox.label(key, "urgent", expected_revision=1)
        inbox.label(key, "top")
        snapshot = json.loads(inbox.state())
        self.assertEqual(snapshot["labels"], 1)
        self.assertEqual(snapshot["cards"][0]["human_urgency"], "top")

    def test_interrupted_burst_keeps_first_completion_and_revision(self):
        slow_started = Event()
        release_slow = Event()

        def model(text, image, rules, **kwargs):
            if text == "second" and len(rules) <= 1:
                if not rules:
                    slow_started.set()
                release_slow.wait(2)
            return outcome("fallback" if not rules or len(rules) == 2 else "zoro")

        self.model.side_effect = model
        inbox = self.open_inbox()
        try:
            inbox.add_examples()
            self.assertTrue(slow_started.wait(1))
            snapshot = self.wait_for(lambda: json.loads(inbox.state()), lambda s: s["burst"]["done"] == 1)
            self.assertEqual(snapshot["burst"]["fallback"], 1)
            inbox.rule("첫 번째 조건")
            self.assertEqual(json.loads(inbox.state())["burst"]["done"], 1)
        finally:
            release_slow.set()
        finished = self.settled(inbox, 2, 1)["burst"]
        self.assertEqual(finished["done"], 2)
        self.assertEqual(finished["fallback"], 1)
        self.assertEqual(finished["started_revision"], 0)
        self.assertEqual(finished["revisions"], [0, 1])

        inbox.rule("두 번째 조건")
        after = self.settled(inbox, 2, 2)
        self.assertEqual(after["burst"], finished)
        self.assertTrue(all(card["decision"]["mode"] == "fallback" for card in after["cards"]))

    def test_restart_recovers_first_completions_in_a_legacy_burst(self):
        self.model.side_effect = lambda text, image, rules, **kwargs: outcome("zoro" if rules else "fallback")
        inbox = self.open_inbox()
        first, waiting = inbox.add_examples()
        self.settled(inbox, 2, 0)
        inbox.rule("조건")
        state = self.settled(inbox, 2, 1)
        inbox.close()

        legacy = {key: value for key, value in state["burst"].items() if key not in ("started_revision", "revisions")}
        legacy.update(done=0, fallback=0)
        with closing(sqlite3.connect(self.home / "inbox.sqlite")) as db, db:
            db.execute("UPDATE meta SET data=? WHERE key='burst'", (json.dumps(legacy),))
            for key, payload in db.execute("SELECT id,data FROM cards").fetchall():
                card = json.loads(payload)
                del card["burst_first_revision"]
                if key == waiting:
                    card.update(status="pending", urgency=None, decision=None)
                db.execute("UPDATE cards SET data=? WHERE id=?", (json.dumps(card), key))
            for seq, payload in db.execute("SELECT seq,payload FROM journal").fetchall():
                if json.loads(payload).get("id") == waiting:
                    db.execute("DELETE FROM journal WHERE seq=?", (seq,))

        reopened = self.open_inbox()
        finished = self.settled(reopened, 2, 1)["burst"]
        self.assertEqual(reopened.cards[first]["burst_first_revision"], 0)
        self.assertEqual(reopened.cards[waiting]["burst_first_revision"], 1)
        self.assertEqual(finished["done"], 2)
        self.assertEqual(finished["fallback"], 1)
        self.assertEqual(finished["revisions"], [0, 1])


if __name__ == "__main__":
    unittest.main()
