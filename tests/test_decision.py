import base64
import io
import json
import os
from threading import Barrier, Event
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError

from PIL import Image

import decision


REPORT = "지금 집에 불이 났어요. 안에 아이가 갇혀 있어요."
with Image.new("RGB", (64, 48)) as image, io.BytesIO() as buffer:
    image.save(buffer, "JPEG")
    PHOTO = buffer.getvalue()
BASE_PERSON = {"life_threat": 0.9, "trapped": 0.05, "minor_injury": 0.025, "none": 0.025}
NO_PERSON = {"life_threat": 0.05, "trapped": 0.05, "minor_injury": 0.05, "none": 0.85}
BASE_PROBS = {
    "people": 1 - BASE_PERSON["none"], "now": 0.9, "life_threat": BASE_PERSON["life_threat"],
    "severe": 0.9, "field_photo": 0.9, "ongoing_hazard": 0.9, "mismatch": None,
}


def model_response(claim=("fire", 0.9), photo=("fire_smoke_burn", 0.9),
                   person=BASE_PERSON, **overrides):
    probs = BASE_PROBS | overrides
    choices = {"claim_kind": claim, "photo_kind": photo}

    def respond(request, timeout):
        body = json.loads(request.data)
        answers = {}
        for key, question in body["questions"].items():
            if key == "person_state":
                # The reported argmax is ignored by the calibrated policy.
                answers[key] = {"type": "choice", "choice": "life_threat", "probabilities": person}
            elif question["type"] == "noul":
                answers[key] = {"noul": probs[key]}
            else:
                kind, probability = choices[key]
                alternatives = question["criteria"].keys() - {kind}
                answers[key] = {
                    "type": "choice",
                    "choice": kind,
                    "probabilities": dict.fromkeys(alternatives, (1 - probability) / len(alternatives))
                    | {kind: probability},
                    "confidence": 0.99,
                }
        return io.BytesIO(json.dumps({"model": "model-revision", "answers": answers}).encode())

    return respond


class DecisionTest(unittest.TestCase):
    def setUp(self):
        threshold = patch.object(decision, "PERSON_THRESHOLD", 0.5)
        threshold.start()
        self.addCleanup(threshold.stop)

    def test_exact_baseline_prompts_and_modalities(self):
        both_started = Barrier(2)
        response = model_response()

        def respond_concurrently(request, timeout):
            both_started.wait(timeout=timeout)
            return response(request, timeout)

        with (
            patch.dict(os.environ, {"ZORO_URL": "http://zoro.example:8080/v1/"}),
            patch.object(decision, "urlopen", side_effect=respond_concurrently) as send,
        ):
            result = decision.decide(REPORT, PHOTO, [])

        self.assertEqual(result["mode"], "zoro")
        self.assertEqual(result["model"], "model-revision")
        self.assertEqual(result["urgency"], "top")
        self.assertEqual(result["probs"], BASE_PROBS)
        self.assertEqual(result["person_state"], "life_threat")
        self.assertEqual(result["person_probabilities"], BASE_PERSON)
        self.assertEqual(result["person_threshold"], decision.PERSON_THRESHOLD)
        self.assertFalse(result["review"])
        self.assertIs(result["mismatch"], False)
        self.assertIs(result["non_field_photo"], False)
        self.assertEqual((result["claim_kind"], result["claim_prob"]), ("fire", 0.9))
        self.assertEqual((result["photo_kind"], result["photo_prob"]), ("fire_smoke_burn", 0.9))
        self.assertGreaterEqual(result["ms"], 0)
        self.assertEqual(send.call_count, 2)

        bodies = {}
        for call in send.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, "http://zoro.example:8080/v1/systemone")
            self.assertEqual(request.method, "POST")
            self.assertLessEqual(call.kwargs["timeout"], decision.DECISION_TIMEOUT)
            self.assertGreater(call.kwargs["timeout"], 0)
            body = json.loads(request.data)
            self.assertEqual(body["model"], "zoro-1")
            bodies[bool(body.get("images"))] = body

        text = bodies[False]
        self.assertNotIn("images", text)
        self.assertEqual(text["state"], {"신고": REPORT})
        self.assertEqual(list(text["questions"]), ["person_state", "now", "claim_kind"])
        self.assertEqual(text["questions"], {
            "person_state": decision.PERSON_QUESTION,
            "now": {"type": "noul", "instructions": "글은 사건이 현재 진행 중이라고 신고하는가? 과거 사진 공유나 문의는 아니오."},
            "claim_kind": decision.TEXT_QUESTION,
        })

        photo = bodies[True]
        self.assertEqual(photo["state"], "첨부 사진")
        self.assertEqual(photo["questions"], {
            "severe": {"type": "noul", "instructions": "사진에 구조물 붕괴, 지붕 소실, 도로 유실 등 심각한 물적 피해가 보이는가?"},
            "field_photo": {"type": "noul", "instructions": decision.FIELD_PHOTO_QUESTION},
            "photo_kind": decision.PHOTO_QUESTION,
            "ongoing_hazard": {"type": "noul", "instructions": decision.QUESTIONS["ongoing_hazard"]},
        })
        self.assertEqual(len(text["questions"]) + len(photo["questions"]), 7)
        self.assertEqual(len(photo["images"]), 1)
        self.assertEqual(base64.b64decode(photo["images"][0], validate=True), PHOTO)

    def test_urgency_rubric(self):
        trapped = {"life_threat": 0.09, "trapped": 0.41, "minor_injury": 0.1, "none": 0.4}
        minor = {"life_threat": 0.02, "trapped": 0.035, "minor_injury": 0.48, "none": 0.465}
        for now, person, severe, hazard, state, urgency in (
            (0.9, NO_PERSON, 0.1, 0.1, "none", "normal"),
            (0.9, NO_PERSON, 0.9, 0.1, "none", "urgent"),
            (0.9, NO_PERSON, 0.1, 0.9, "none", "urgent"),
            (0.9, trapped, 0.9, 0.49, "trapped", "urgent"),
            (0.9, trapped, 0.1, 0.50, "trapped", "top"),
            (0.9, minor, 0.9, 0.9, "minor_injury", "urgent"),
            (0.9, BASE_PERSON, 0.1, 0.1, "life_threat", "top"),
            (0.1, BASE_PERSON, 0.9, 0.49, "life_threat", "non_urgent"),
            (0.1, BASE_PERSON, 0.9, 0.50, "life_threat", "urgent"),
        ):
            response = model_response(now=now, person=person, severe=severe, ongoing_hazard=hazard)
            with (
                self.subTest(now=now, state=state, severe=severe, hazard=hazard),
                patch.object(decision, "urlopen", side_effect=response),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertEqual(result["person_state"], state)
                self.assertEqual(result["urgency"], urgency)
                self.assertEqual(result["probs"]["people"], 1 - person["none"])
                self.assertEqual(result["probs"]["life_threat"], person["life_threat"])
                self.assertEqual(result["probs"]["ongoing_hazard"], hazard)
                self.assertEqual(result["review"], state in ("trapped", "minor_injury") or (now < 0.5 <= hazard))

    def test_threshold_uses_absolute_life_probability_and_next_best_class(self):
        for person, state, urgency in (
            ({"life_threat": 0.30, "trapped": 0.05, "minor_injury": 0.05, "none": 0.60}, "life_threat", "top"),
            ({"life_threat": 0.29, "trapped": 0.05, "minor_injury": 0.06, "none": 0.60}, "none", "normal"),
            ({"life_threat": 0.29, "trapped": 0.22, "minor_injury": 0.21, "none": 0.28}, "none", "normal"),
        ):
            response = model_response(person=person, severe=0.1, ongoing_hazard=0.1)
            with (
                self.subTest(person=person),
                patch.object(decision, "PERSON_THRESHOLD", 0.30),
                patch.object(decision, "urlopen", side_effect=response),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertEqual(result["person_state"], state)
                self.assertEqual(result["urgency"], urgency)

    def test_past_and_visible_hazard_force_review_even_with_a_matched_rule(self):
        response = model_response(now=0.1, person=NO_PERSON, severe=0.1, ongoing_hazard=0.9, rule_1=0.9)
        with patch.object(decision, "urlopen", side_effect=response):
            result = decision.decide(REPORT, PHOTO, ["고립된 고령자·어린이는 최우선"])
        self.assertEqual(result["urgency"], "urgent")
        self.assertTrue(result["review"])
        self.assertTrue(any("지난 사건" in reason and "사진" in reason for reason in result["reasons"]))
        self.assertTrue(result["rules"][0]["matched"])
        self.assertFalse(result["rules"][0]["applied"])

    def test_only_supported_conflicts_above_threshold_trigger_mismatch_review(self):
        for claim, photo, mismatch in (
            (("fire", 0.8), ("water_storm", 0.8), True),
            (("quake_collapse", 0.95), ("fire_smoke_burn", 0.95), True),
            (("fire", 0.79), ("water_storm", 0.95), False),
            (("fire", 0.95), ("collapse", 0.79), False),
            (("storm_flood", 0.95), ("collapse", 0.95), False),
            (("fire", 0.95), ("unclear", 0.95), False),
            (("other", 0.95), ("fire_smoke_burn", 0.95), False),
            (("fire", 0.6), ("fire_smoke_burn", 0.6), False),
        ):
            with (
                self.subTest(claim=claim, photo=photo),
                patch.object(decision, "MISMATCH_THRESHOLD", 0.8),
                patch.object(decision, "urlopen", side_effect=model_response(
                    claim=claim, photo=photo, now=0.1, ongoing_hazard=0.1,
                )),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertIs(result["mismatch"], mismatch)
                self.assertIs(result["review"], mismatch)
                self.assertEqual(result["urgency"], "non_urgent")
                self.assertIsNone(result["probs"]["mismatch"])

    def test_core_uncertainty_threshold_is_inclusive(self):
        for answers in ({"now": 0.35}, {"severe": 0.65}, {"person": {"life_threat": 0.2, "trapped": 0.1, "minor_injury": 0.1, "none": 0.6}}):
            with (
                self.subTest(answers=answers),
                patch.object(decision, "urlopen", side_effect=model_response(**answers)),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertTrue(result["review"])
                self.assertIs(result["mismatch"], False)

    def test_non_field_photo_triggers_review_below_point_five(self):
        for probability, flagged in ((0.0, True), (0.49, True), (0.5, False), (0.55, False), (1.0, False)):
            with (
                self.subTest(probability=probability),
                patch.object(decision, "urlopen", side_effect=model_response(field_photo=probability)),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertEqual(result["mode"], "zoro")
                self.assertEqual(result["probs"]["field_photo"], probability)
                self.assertIs(result["non_field_photo"], flagged)
                self.assertIs(result["review"], flagged)
                self.assertEqual(result["urgency"], "top")

    def test_rule_decision_updates_an_ongoing_report_only(self):
        rules = ["지하 통로 출입구가 잠김", "유치원에 피해가 보임"]
        for now, expected in ((0.9, "top"), (0.1, "non_urgent")):
            response = model_response(person=NO_PERSON, severe=0.1, now=now,
                                      ongoing_hazard=0.1, rule_1=0.9, rule_2=0.43)
            with self.subTest(now=now), patch.object(decision, "urlopen", side_effect=response) as send:
                result = decision.decide(REPORT, PHOTO, rules)
            self.assertEqual(result["urgency"], expected)
            self.assertEqual(result["probs"]["rule_1"], 0.9)
            self.assertEqual(result["probs"]["rule_2"], 0.43)
            self.assertFalse(result["review"])
            self.assertEqual(result["rules"], [
                {"key": "rule_1", "condition": rules[0], "matched": True, "applied": now >= 0.5},
                {"key": "rule_2", "condition": rules[1], "matched": False, "applied": False},
            ])
            bodies = [json.loads(call.args[0].data) for call in send.call_args_list]
            self.assertEqual(len(bodies), 3)
            rule_request = next(body for body in bodies if "rule_1" in body["questions"])
            self.assertEqual(rule_request["state"], {"신고": REPORT})
            self.assertEqual(rule_request["images"], [base64.b64encode(PHOTO).decode("ascii")])
            questions = rule_request["questions"]
            self.assertEqual(set(questions), {"rule_1", "rule_2"})
            for key, condition in zip(("rule_1", "rule_2"), rules):
                self.assertEqual(questions[key]["type"], "noul")
                self.assertIn(condition, questions[key]["instructions"])

    def test_demo_and_explicit_modality_rules_share_the_two_requests(self):
        rules = ["고립된 고령자·어린이는 최우선", "사진: 도로가 침수됨", "글: 임산부가 고립됨"]
        response = model_response(person=NO_PERSON, severe=0.1, ongoing_hazard=0.1,
                                  rule_1=0.9, rule_2=0.3, rule_3=0.2)
        with patch.object(decision, "urlopen", side_effect=response) as send:
            result = decision.decide(REPORT, PHOTO, rules)

        self.assertEqual(result["urgency"], "top")
        self.assertEqual([rule["condition"] for rule in result["rules"]], rules)
        self.assertEqual(send.call_count, 2)
        bodies = [json.loads(call.args[0].data) for call in send.call_args_list]
        text = next(body for body in bodies if "images" not in body)
        photo = next(body for body in bodies if "images" in body)
        self.assertEqual(text["state"], {"신고": REPORT})
        self.assertEqual(photo["state"], "첨부 사진")
        self.assertEqual(set(text["questions"]) & {"rule_1", "rule_2", "rule_3"}, {"rule_1", "rule_3"})
        self.assertEqual(set(photo["questions"]) & {"rule_1", "rule_2", "rule_3"}, {"rule_2"})
        self.assertIn("우선 처리 기준: 도로가 침수됨", photo["questions"]["rule_2"]["instructions"])
        self.assertIn("우선 처리 기준: 임산부가 고립됨", text["questions"]["rule_3"]["instructions"])

    def test_simulated_outage_uses_keywords_without_fabricating_evidence(self):
        with patch.object(decision, "urlopen") as send:
            result = decision.decide(REPORT, PHOTO, ["불이 남"], force_fallback=True)
        send.assert_not_called()
        self.assertEqual(result["mode"], "fallback")
        self.assertEqual(result["urgency"], "urgent")
        self.assertEqual(result["probs"], dict.fromkeys((*BASE_PROBS, "rule_1")))
        self.assertTrue(result["review"])
        for key in ("mismatch", "claim_kind", "claim_prob", "photo_kind", "photo_prob", "non_field_photo",
                    "person_state", "person_probabilities"):
            self.assertIsNone(result[key])
        self.assertIn("모의", result["reasons"][0])
        self.assertTrue(any("잠정" in reason for reason in result["reasons"]))
        self.assertTrue(any("사진" in reason and "진행 위험" in reason for reason in result["reasons"]))
        self.assertEqual(result["rules"], [
            {"key": "rule_1", "condition": "불이 남", "matched": None, "applied": False},
        ])

    def test_fallback_handles_past_and_negated_people_terms(self):
        for text, expected in (
            ("지난주 화재 사진을 공유합니다. 당시 다친 사람이 있었습니다.", "non_urgent"),
            ("지금 침수됐지만 다친 사람은 없습니다.", "normal"),
            ("다친 사람은 없지만 아직 차에 아이가 갇혔어요.", "urgent"),
            ("어제부터 침수됐고 지금 아이가 고립됐어요. 사진을 보냅니다.", "urgent"),
        ):
            with self.subTest(text=text):
                self.assertEqual(decision.decide(text, PHOTO, [], force_fallback=True)["urgency"], expected)

    def test_connection_failure_and_invalid_answers_use_fallback(self):
        with patch.object(decision, "urlopen", side_effect=URLError(ConnectionRefusedError())):
            result = decision.decide(REPORT, PHOTO, [])
        self.assertEqual(result["mode"], "fallback")
        self.assertIn("연결", result["reasons"][0])

        for value in (-0.1, 1.1, float("nan"), "0.9", True):
            with (
                self.subTest(value=value),
                patch.object(decision, "urlopen", side_effect=model_response(now=value)),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertEqual(result["mode"], "fallback")
                self.assertIsNone(result["probs"]["mismatch"])

        with patch.object(decision, "urlopen", side_effect=model_response(field_photo=True)):
            result = decision.decide(REPORT, PHOTO, [])
        self.assertEqual(result["mode"], "fallback")
        self.assertIsNone(result["probs"]["field_photo"])
        self.assertIsNone(result["non_field_photo"])

        response = model_response()

        criteria = decision.PHOTO_QUESTION["criteria"]
        for probabilities in (list(criteria), dict.fromkeys(criteria, True)):
            def invalid_choice(request, timeout):
                result = json.load(response(request, timeout))
                if "photo_kind" in result["answers"]:
                    result["answers"]["photo_kind"]["probabilities"] = probabilities
                return io.BytesIO(json.dumps(result).encode())

            with (
                self.subTest(probabilities=probabilities),
                patch.object(decision, "urlopen", side_effect=invalid_choice),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertEqual(result["mode"], "fallback")
                self.assertIsNone(result["mismatch"])

    def test_incomplete_or_invalid_person_distribution_uses_fallback(self):
        for person in (
            {"life_threat": 0.9, "none": 0.1},
            BASE_PERSON | {"unexpected": 0.0},
            BASE_PERSON | {"trapped": True},
            BASE_PERSON | {"trapped": float("nan")},
            BASE_PERSON | {"trapped": -0.05},
            BASE_PERSON | {"none": 0.5},
        ):
            with (
                self.subTest(person=person),
                patch.object(decision, "urlopen", side_effect=model_response(person=person)),
            ):
                result = decision.decide(REPORT, PHOTO, [])
                self.assertEqual(result["mode"], "fallback")
                self.assertIsNone(result["person_probabilities"])

    def test_end_to_end_timeout_does_not_wait_for_transport(self):
        release = Event()
        respond = model_response()

        def stalled_response(request, timeout):
            release.wait(1)
            return respond(request, timeout)

        with (
            patch.object(decision, "DECISION_TIMEOUT", 0.05),
            patch.object(decision, "urlopen", side_effect=stalled_response) as send,
        ):
            try:
                started = time.monotonic()
                result = decision.decide(REPORT, PHOTO, [])
                elapsed = time.monotonic() - started
            finally:
                release.set()

        self.assertGreater(send.call_count, 0)
        self.assertLess(elapsed, 0.4)
        self.assertEqual(result["mode"], "fallback")
        self.assertIn("시간 초과", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
