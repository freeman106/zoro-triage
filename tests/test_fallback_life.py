import unittest

import decision


class FallbackLifeThreatTest(unittest.TestCase):
    def test_explicit_life_risk_is_top_and_requires_review(self):
        for report in (
            "아이가 숨을 안 쉼. 빨리 와주세요.",
            "넘어진 사람이 숨을 못 쉬어요.",
            "환자가 숨을 쉬지 않습니다.",
            "숨을 쉬지 않아요.",
            "숨을 쉴 수 없어요.",
            "피가 많이 흐르고 있어요.",
            "환자의 호흡이 멎었어요.",
            "동료가 호흡곤란을 호소해요.",
            "환자가 호흡이 곤란합니다.",
            "환자 의식 없음.",
            "사람이 의식을 잃었어요.",
            "무너진 벽에 사람이 깔렸습니다.",
            "잔해 아래에 깔린 사람을 발견했어요.",
            "자동차 아래에 작업자 깔림.",
            "건물 잔해에 깔림",
            "차에 깔렸어요",
            "깔림",
            "사람이 깔린 상태예요",
            "시민이 깔려 있는 상태입니다.",
            "아이가 건물 잔해에 깔려 있는 상태입니다.",
            "심한 출혈이 있어요.",
            "다량 출혈",
            "출혈이 심합니다.",
            "다리에서 피가 많이 나요.",
            "지난주 화재 사진을 공유합니다. 당시 환자가 호흡을 하지 못했어요.",
        ):
            with self.subTest(report=report):
                result = decision.decide(report, b"photo", ["어린이 우선"], force_fallback=True)
                self.assertEqual(result["mode"], "fallback")
                self.assertEqual(result["urgency"], "top")
                self.assertTrue(result["review"])
                self.assertTrue(all(value is None for value in result["probs"].values()))
                self.assertTrue(any("생명 위협" in reason for reason in result["reasons"]))

    def test_negation_and_unrelated_keywords_do_not_raise_urgency(self):
        for report in (
            "바닥에 깔린 전선과 매트가 젖었어요.",
            "운전자 없는 차량이 건물에 깔렸어요.",
            "사람이 깔렸다는 말은 사실이 아닙니다.",
            "지금 사람이 지나가는 바닥에 깔린 매트가 젖었어요.",
            "잔해에 깔린 전선을 치워주세요.",
            "잔해에 깔려 있는 전선을 치워주세요.",
            "짐이 차에 깔렸어요.",
            "차에 물건이 깔렸어요.",
            "철판에 매트가 깔림",
            "깔림 방지 안내판 사진입니다.",
            "차에 깔림 없음.",
            "사람은 의식이 있고 숨을 잘 쉽니다. 출혈 없음.",
            "숨을 못 쉬는 건 아니에요.",
            "환자는 호흡곤란이 없고 출혈이 심하지 않아요.",
            "호흡이 곤란하지 않아요.",
            "의식을 잃은 환자는 없습니다.",
            "심한 출혈은 없습니다.",
            "깔린 사람은 아무도 없어요.",
            "사람 깔림 아님.",
            "피를 많이 흘리지 않았어요.",
        ):
            with self.subTest(report=report):
                result = decision.decide(report, b"", [])
                self.assertEqual(result["urgency"], "normal")

    def test_isolation_alone_is_urgent_and_life_threat_has_precedence(self):
        isolated = decision.decide("지금 아이가 고립됐어요.", b"", [])
        self.assertEqual(isolated["urgency"], "urgent")
        trapped = decision.decide("차 안에 꼼짝없이 갇힘.", b"", [])
        self.assertEqual(trapped["urgency"], "urgent")

        result = decision.decide("호흡곤란은 없지만 출혈이 심해요.", b"", [])
        self.assertEqual(result["urgency"], "top")
