"""Exercise a private demo119 instance with an empty, dedicated data directory."""

import argparse
import base64
import json
import os
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from person_choice import choose as choose_person, validate as validate_person


LEVELS = ("top", "urgent", "normal", "non_urgent")
DEFAULT_IMAGE_ROOT = "/ephemeral/crisis_bench/crisismmd"
RULE = "고립된 고령자·어린이는 최우선"
MODEL_PROBS = ("people", "now", "severe", "field_photo", "life_threat", "ongoing_hazard")
EXAMPLE_ORDER = ("top", "mismatch", "outage", "past_hazard", "rule")
BURST_COUNT = 100


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


class Demo:
    def __init__(self, url, admin_token, report_token, timeout):
        self.url = url.rstrip("/")
        self.admin_token = admin_token
        self.report_token = report_token
        self.timeout = timeout

    def open(self, path, token=None, data=None, timeout=20):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = None
        if data is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return urlopen(Request(self.url + path, data=payload, headers=headers), timeout=timeout)

    def request(self, path, token=None, data=None, denied=False):
        try:
            with self.open(path, token, data) as response:
                if denied:
                    raise RuntimeError(f"{path} accepted unauthorized access")
                raw = response.read()
                return json.loads(raw) if raw else None
        except HTTPError as error:
            with error:
                if denied and error.code in (401, 403):
                    return error.code
                raise RuntimeError(f"{path} returned HTTP {error.code}") from error

    def admin(self, path, data=None):
        return self.request(path, self.admin_token, data)

    def state(self):
        return self.admin("/api/state")

    def wait(self, predicate, stage):
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            snapshot = self.state()
            if predicate(snapshot):
                return snapshot, elapsed(start)
            time.sleep(0.35)
        raise RuntimeError(f"{stage} timed out after {self.timeout:g}s")

    def stream(self, events):
        try:
            with self.open("/api/events", self.admin_token, timeout=10) as response:
                check("text/event-stream" in response.headers.get("Content-Type", ""),
                      "board events did not open an SSE stream")
                for line in response:
                    if line.startswith(b"data:"):
                        events.put((time.monotonic(), time.time(), json.loads(line[5:])))
        except (OSError, ValueError, RuntimeError) as error:
            events.put(error)


def next_event(events):
    try:
        event = events.get(timeout=10)
    except Empty as error:
        raise RuntimeError("board SSE state event timed out") from error
    if isinstance(event, Exception):
        raise RuntimeError(f"board SSE failed: {event}") from event
    return event


def elapsed(start):
    return round((time.monotonic() - start) * 1000, 3)


def find(snapshot, report_id):
    return next((card for card in snapshot["cards"] if card["id"] == report_id), None)


def done(snapshot, count, revision):
    return len(snapshot["cards"]) == count and all(
        card["status"] == "done" and card["rule_revision"] == revision
        for card in snapshot["cards"]
    )


def rank(card):
    if card["status"] == "pending":
        return (0, 0, False, card["created_at"])
    review = (card.get("reason_review") or {}).get("needs_review", card["decision"]["review"])
    needs_review = card["human_urgency"] is None and review
    return (
        1,
        LEVELS.index(card["human_urgency"] or card["urgency"]),
        not needs_review,
        card["created_at"],
    )


def assert_sorted(snapshot):
    order = [rank(card) for card in snapshot["cards"]]
    check(order == sorted(order), "board sort order disagrees with urgency, review and receipt order")


def decision_result(card):
    decision = card["decision"]
    return {
        "id": card["id"],
        "urgency": card["urgency"],
        **{key: decision[key] for key in (
            "mode", "review", "mismatch", "non_field_photo", "claim_kind", "claim_prob",
            "photo_kind", "photo_prob", "probs", "person_state", "person_probabilities", "person_threshold",
        )},
    }


def prove(demo, fixture_root, image_root, metrics):
    rows = json.loads((fixture_root / "reports.json").read_text(encoding="utf-8"))
    examples = {row["example"]: row for row in rows if "example" in row}
    check(len(rows) == BURST_COUNT, f"expected {BURST_COUNT} fixture reports")
    check(set(examples) == set(EXAMPLE_ORDER), "unexpected example fixtures")
    example_count = 1 + len(EXAMPLE_ORDER)
    full_count = example_count + BURST_COUNT
    photo = (image_root / examples["top"]["image_path"]).read_bytes()
    phone = {"image": base64.b64encode(photo).decode("ascii"), "text": examples["top"]["text"]}

    initial = demo.state()
    check(initial["cards"] == [] and initial["labels"] == 0 and initial["revision"] == 0,
          "use an isolated demo instance with empty state")
    metrics["auth"] = {
        "anonymous_state": demo.request("/api/state", denied=True),
        "report_state": demo.request("/api/state", demo.report_token, denied=True),
        "report_events": demo.request("/api/events", demo.report_token, denied=True),
        "report_rule": demo.request("/api/rules", demo.report_token, {"text": RULE}, denied=True),
        "anonymous_upload": demo.request("/api/reports", data=phone, denied=True),
    }

    events = Queue()
    threading.Thread(target=demo.stream, args=(events,), daemon=True).start()
    _, _, first = next_event(events)
    check(first["cards"] == [], "first SSE state was not empty")
    start = time.monotonic()
    receipt = demo.request("/api/reports", demo.report_token, phone)
    receipt_ms = elapsed(start)
    phone_id = receipt["id"]
    check(bool(phone_id), "phone receipt had no ID")
    while True:
        seen_at, seen_wall, snapshot = next_event(events)
        received = find(snapshot, phone_id)
        if received is not None:
            break
    check(received["text"] == phone["text"] and received["source"] == "phone",
          "SSE card did not match the phone submission")
    assert_sorted(snapshot)
    with demo.open(received["image_url"], demo.admin_token) as response:
        thumbnail = response.read()
        check(response.headers.get_content_type() == "image/jpeg"
              and thumbnail.startswith(b"\xff\xd8") and thumbnail.endswith(b"\xff\xd9"),
              "board photo did not return a JPEG preview")
    metrics["phone"] = dict(id=phone_id, receipt_ms=receipt_ms,
                            upload_to_sse_ms=round((seen_at - start) * 1000, 3),
                            server_receipt_to_sse_ms=round((seen_wall - received["created_at"]) * 1000, 3))

    start = time.monotonic()
    decisions = {}
    for count, kind in enumerate(EXAMPLE_ORDER, start=2):
        example_id = demo.admin("/api/example", {"kind": kind})["id"]
        baseline, _ = demo.wait(lambda s: done(s, count, 0), f"{kind} example")
        assert_sorted(baseline)
        matched = find(baseline, example_id)
        check(matched and matched["source"] == "example" and matched["scenario_id"] == examples[kind]["id"],
              f"receipt did not match {kind} example")
        decisions[kind] = matched
    metrics["examples"] = dict(
        elapsed_ms=elapsed(start),
        results={key: decision_result(card) for key, card in decisions.items()},
    )
    top, mismatch, outage, rule, past = (
        metrics["examples"]["results"][key] for key in ("top", "mismatch", "outage", "rule", "past_hazard")
    )
    for name, result in metrics["examples"]["results"].items():
        if result["mode"] != "zoro":
            continue
        person_probabilities = validate_person(result["person_probabilities"])
        threshold = result["person_threshold"]
        check(type(threshold) in (int, float) and 0 <= threshold <= 1,
              f"{name} example did not log a valid life-threat threshold")
        check(result["person_state"] == choose_person(person_probabilities, threshold)
              and result["probs"]["people"] == 1 - person_probabilities["none"]
              and result["probs"]["life_threat"] == person_probabilities["life_threat"],
              f"{name} example person-state evidence disagreed with the calibrated policy")
    top_probs = top["probs"]
    check(top["urgency"] == "top" and top["mode"] == "zoro" and top_probs["now"] >= 0.5
          and (top["person_state"] == "life_threat"
               or (top["person_state"] == "trapped" and top_probs["ongoing_hazard"] >= 0.5)),
          "top example did not meet the live Zoro top policy")
    check(mismatch["mode"] == "zoro" and mismatch["mismatch"] is True and mismatch["review"] is True,
          "mismatch example did not enter review with a positive mismatch")
    check(outage["mode"] == "fallback" and outage["urgency"] == "top" and outage["review"] is True
          and set(MODEL_PROBS).issubset(outage["probs"])
          and all(prob is None for prob in outage["probs"].values())
          and outage["person_state"] is None and outage["person_probabilities"] is None,
          "outage example did not enter top fallback review with unavailable model probabilities")
    check(rule["urgency"] in ("normal", "urgent") and rule["mode"] == "zoro",
          "rule example was already top, past, or unavailable")
    check(past["mode"] == "zoro" and past["probs"]["now"] < 0.5
          and past["probs"]["ongoing_hazard"] >= 0.5 and past["urgency"] == "urgent"
          and past["review"] is True,
          "past hazard example did not enter urgent review with conflicting text and photo signals")

    recovery = decisions["outage"]["recovery"]
    check(recovery["status"] == "waiting_for_outage", "simulated outage did not wait for recovery toggle")
    demo.admin("/api/outage-recovery", {"enabled": True})
    recovered_state, _ = demo.wait(
        lambda state: (find(state, outage["id"]).get("recovery") or {}).get("status") == "recovered",
        "automatic Zoro retry",
    )
    recovered = find(recovered_state, outage["id"])
    retry = recovered["recovery"]
    check(recovered["decision"]["mode"] == "zoro" and retry["attempted_at"] >= recovery["due_at"],
          "automatic retry did not return real evidence after the wait")
    metrics["outage_recovery"] = {
        "id": outage["id"], "first_mode": outage["mode"], "final_mode": recovered["decision"]["mode"],
        "status": retry["status"], "due_at": recovery["due_at"], "attempted_at": retry["attempted_at"],
        "finished_at": retry["finished_at"], "final_urgency": recovered["urgency"],
    }

    demo.admin("/api/reports/" + phone_id + "/label", {"urgency": "non_urgent"})
    before = demo.state()
    check(before["labels"] == 1 and find(before, phone_id)["human_urgency"] == "non_urgent",
          "operator override or label count did not survive a state read")
    check(1 < len(before["cards"]) <= 10 and before["burst"] is None,
          "rule demo must run on at most ten cards before the burst")
    assert_sorted(before)

    start = time.monotonic()
    demo.admin("/api/rules", {"text": RULE})
    after, _ = demo.wait(lambda s: done(s, example_count, 1), "rule re-evaluation")
    check(after["revision"] == 1 and after["rules"] == [RULE], "rule revision was not recorded")
    before_ids = [c["id"] for c in before["cards"]]
    after_ids = [c["id"] for c in after["cards"]]
    check(set(before_ids) == set(after_ids), "rule re-evaluation changed card identities")
    promoted = find(after, rule["id"])
    previous_rank = before_ids.index(rule["id"]) + 1
    new_rank = after_ids.index(rule["id"]) + 1
    control_rank_before = before_ids.index(past["id"]) + 1
    control_rank_after = after_ids.index(past["id"]) + 1
    metrics["rule"] = dict(revision=1, recomputed=len(after_ids), elapsed_ms=elapsed(start),
                           before=rule["urgency"], after=promoted["urgency"],
                           promoted=decision_result(promoted),
                           rank_before=previous_rank, rank_after=new_rank,
                           control=dict(id=past["id"], rank_before=control_rank_before,
                                        rank_after=control_rank_after))
    check(promoted["urgency"] == "top" and promoted["decision"]["mode"] == "zoro"
          and any(r["condition"] == RULE and r["applied"] for r in promoted["decision"]["rules"]),
          "live rule did not promote the existing card to top")
    check(new_rank < previous_rank, "promoted example did not move upward in the board")
    check(control_rank_before < previous_rank and new_rank < control_rank_after
          and find(after, past["id"])["urgency"] == "urgent",
          "promoted example did not pass the unchanged urgent past-hazard control")
    assert_sorted(after)
    check(after["labels"] == 1 and find(after, phone_id)["human_urgency"] == "non_urgent",
          "rule re-evaluation overwrote the operator label")
    metrics["labels"] = dict(count=after["labels"], persisted_after_rule=True)

    start = time.monotonic()
    burst_ids = demo.admin("/api/burst", {})["ids"]
    completed, _ = demo.wait(lambda s: done(s, full_count, 1), f"{BURST_COUNT}-report burst")
    burst = completed["burst"]
    batch = [card for card in completed["cards"] if card["source"] == "burst"]
    fallback = sum(card["decision"]["mode"] == "fallback" for card in batch)
    metrics["burst"] = dict(count=len(batch), elapsed_ms=burst["elapsed_ms"],
                            observed_ms=elapsed(start), fallback=burst["fallback"], current_fallback=fallback,
                            started_revision=burst["started_revision"], revisions=burst["revisions"])
    check(len(batch) == burst["total"] == burst["done"] == BURST_COUNT,
          f"burst did not complete all {BURST_COUNT} reports")
    check(len(burst_ids) == BURST_COUNT and {c["id"] for c in batch} == set(burst_ids),
          "burst receipt IDs differ")
    check({c["scenario_id"] for c in batch} == {r["id"] for r in rows}, "burst scenario IDs differ")
    check(fallback <= burst["fallback"], "current fallback count exceeds immutable first-pass count")
    check(burst["started_revision"] == 1 and burst["revisions"] == [1],
          "burst did not use the new operating rule throughout")
    assert_sorted(completed)

    check(0 <= metrics["phone"]["server_receipt_to_sse_ms"] <= 300,
          "server receipt to board SSE exceeded 300 ms or clocks differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default=os.getenv("DEMO_URL", "http://127.0.0.1:8119"))
    parser.add_argument("--timeout", type=float, default=180, help="seconds per processing stage")
    args = parser.parse_args()
    metrics = {"passed": False}
    try:
        admin_token = os.environ["ADMIN_TOKEN"]
        report_token = os.environ["REPORT_TOKEN"]
        check(admin_token and report_token and admin_token != report_token, "use two distinct nonempty tokens")
        demo = Demo(args.url, admin_token, report_token, args.timeout)
        prove(demo, Path(__file__).resolve().parent / "fixtures",
              Path(os.getenv("CRISIS_ROOT", DEFAULT_IMAGE_ROOT)), metrics)
        metrics["passed"] = True
    except (OSError, URLError, KeyError, ValueError, RuntimeError) as error:
        metrics["error"] = str(error)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
