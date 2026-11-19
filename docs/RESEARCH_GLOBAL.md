# 해외 도입 근거 조사 (2026-11-19, 조사 에이전트 결과 요약)

표기: [독립] 언론·학술·정부, [회사 주장] 기업 자료, [2차] 원문 미확인. 미확인은 그대로 표시.

## 발표용 핵심 근거
1. **AI는 사람보다 민감하지만 경보만으로는 성과가 안 올랐다.** 코펜하겐 RCT: AI 심정지 탐지 민감도 85.0% vs 접수요원 77.5%, 양성예측도 17.8% vs 55.8%. AI 경보를 받은 요원의 인지율 93.1% vs 90.5% (p=.15, 유의하지 않음). 접수요원은 심정지의 약 25%를 놓침. [독립·학술] JAMA Netw Open 2021-01-04 https://pubmed.ncbi.nlm.nih.gov/33404620/
2. **인력 부족.** 신고센터 74% 결원(2024년 82%), 신입 22% 교육 탈락, 86%가 AI 접수 보조 수용, 63%가 비응급 비중 50~80%라고 응답. 1,379명 응답. NENA·Carbyne 공동 조사(벤더 관여). 2025-06-23 https://carbyne.com/resources/press/third-annual-pulse-of-9-1-1-report-finds-burnout-cyber-vulnerabilities-and-outdated-technology-threaten-emergency-service-readiness/
3. **재난 시 폭주.**
   - 마우이 산불 당일(2023-08-08) 4,523건, 평시 약 3배, 민간 접수요원 자리는 절반도 안 찼음. [독립] https://mauinow.com/2025/08/01/maui-police-department-releases-final-after-action-report-on-aug-8-2023-wildfires/
   - 하비(2017) 휴스턴 약 2.5일간 75,000건(평시 8배 이상). https://www.cbsnews.com/news/houston-flooding-911-calls-after-harvey
   - 해리스 카운티는 평시 하루 3,000건을 시간당 3,000건 받았고, 하루 평균 응답 대기가 22분인 날도 있었음. https://www.khou.com/article/news/investigations/could-problems-calling-911-during-harvey-happen-again-today/285-589328501
   - 텍사스 커 카운티 홍수(2025-07-04): 접수요원 2명이 400건 이상 처리, 사망 117명. AP 2025-12-06 https://www.ksl.com/article/51415370/
4. **EU AI Act가 이 용도를 고위험으로 명시.** 부속서 III 5(d): 응급 신고 평가·분류·출동 우선순위 결정. 제14조(4): 자동화 편향 인식(b), 출력 무시·번복(d), 중지(e). 의무 적용 2027-12-02로 연기(Reg. (EU) 2026/1744). https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3 · https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-14 · https://www.lewissilkin.com/insights/2026/07/27/the-digital-omnibus-on-ai-enters-into-force-today-102nedo
5. **대형 인수.**
   - Axon: Prepared 잔여 99.2%를 약 7.28억 달러에 인수(2025-10), Carbyne 잔여 89.3%를 약 5.61억 달러에 인수(2026-02). [공시] https://www.sec.gov/Archives/edgar/data/0001069183/000162828026031542/axon-20260331.htm
   - Motorola: Hyper 인수(2026-04, 금액 비공개).

## 도입 사례
- **Corti (덴마크)**
  - 2019 후향 연구: AI 민감도 84.1% vs 요원 72.5%, 양성예측도 20.9% vs 33%.
  - 시애틀 소방청은 2023-12부터 모든 의료 신고에 적용, 연 약 26만 달러. 간호사 상담선 이관이 32% 늘었고(Corti는 50%라고 했다가 정정), 공개된 성과 지표는 없음. https://www.govtech.com/em/safety/questions-arise-as-ai-analyzes-redirects-seattle-911-calls
- **Prepared (현 Axon 911)**
  - 누적 투자 약 5,700만 달러, 49개 주 약 1,000개 기관. https://techcrunch.com/2024/09/26/prepared-which-lets-911-dispatchers-text-and-video-chat-with-callers-raises-27m
  - 갈트 경찰서: 연 3만 건 중 73% 이상이 비응급이라고 함. [회사 주장]
- **Carbyne**
  - 2025-08 1억 달러 유치. https://en.globes.co.il/en/article-1001517862
  - 뉴올리언스에서 중복 신고 AI를 약 3년간 비공개로 운영, OECD AI 모니터가 "위험 요소"로 분류.
- **RapidSOS HARMONY** (2024-05 베타): 감성 분석, 번역, 키워드 경보, "영상 객체 탐지"를 발표. 실배포와 정확도는 미확인.
- **비응급 음성 AI**
  - 몬터레이 카운티: 9,635건 중 30.31%를 사람 없이 해결, 월 1,000달러 이하. [정부] https://ntia.gov/category/next-generation-911/improving-911-operations-with-artificial-intelligence
  - 포트워스: 26% 처리. 세일럼: 60.7% 처리(기존 비응급 대기 최대 25분).
- **HighWind (프랑스)**: 2019년 니스 창업. 사진 외상 "사전 진단"과 FR-ALERT 연동을 주장. 모두 회사 주장이고 독립 검증·배포 규모는 미확인.
- **사진·영상 AI 우선순위 판정**
  - 실제 배포가 확인된 건 수신 기능입니다: iOS 18 Emergency SOS Live Video, Android Emergency Live Video(2025-12), Carbyne 영상 수신.
  - 신고자 사진을 AI로 분석해 우선순위를 매기는 기능은 운영 배포를 입증하는 독립·정부 자료를 찾지 못했습니다.

## 과소 분류 연구
- 덴마크(2014): 고긴급 외로 분류된 94,488명 중 당일 사망 152명(0.16%). 그중 18명은 잠재적으로 예방 가능, 확실히 예방 가능은 0명. 18명 중 13명은 프로토콜 미사용·오사용. https://pubmed.ncbi.nlm.nih.gov/25524363/
- 네덜란드 외상(2022, 114,459명): 전문 외상 치료가 필요한 환자 중 최고 우선순위 비율 83.8%, 하위군별 과소 분류 22.7~65.5%. https://pubmed.ncbi.nlm.nih.gov/34019106/
- 영국 검시관 재발 방지 보고서: 수요 폭주로 999 지연(2025-07), 2등급 평균 도착 50분(목표 18분).

## 비용·시장
- 기관별 공개 비용: 그랜드트래버스 첫해 6만 달러 이후 연 7.2만 달러, 윈터가든 연 5.5만 달러, 레이크 카운티 4.5만 달러, 데인 카운티 15.5만 달러(주 보조금 90%), 출라비스타 4년 약 100만 달러(Axon 계약), 시애틀 Corti 연 약 26만 달러.
- CAD 시장 2024년 23억 → 2030년 43억 달러(유료 리서치 추정, 편차 큼). AI 디스패치만 따로 집계한 수치는 없음.

## 규제와 우리 설계
- 사람 최종 판단은 제14조(4)(d), "확인 필요" 배지는 14(4)(b)(c)에 대응합니다.
- 공공기관 배치자는 기본권 영향평가 대상입니다.

## 불리한 자료
1. AI 보조가 사람 성과를 올렸다는 RCT 근거가 없습니다(코펜하겐 무효과, 오경보 많음).
2. 사진 기반 중증도 AI는 HighWind와 RapidSOS가 이미 주장했습니다. "최초"라는 주장은 금지.
3. 사진·영상 수신 채널은 Apple·Google → Axon·RapidSOS·Motorola가 쥐고 있습니다.
4. 공개·형평성 반발 사례가 있습니다(시애틀 Corti, 뉴올리언스 Carbyne 비공개 운영).
5. 음성인식 오류율 편향 연구가 있습니다(흑인 화자 0.35 vs 백인 0.19, PNAS 2020).
6. 과소 분류로 확실히 예방 가능했던 사망은 덴마크 연구에서 0건이었습니다. 과장 금지.
7. 하비 때 문자 신고는 몇 건 수준이었습니다. 멀티미디어 신고 수요가 아직 작을 수 있습니다.
8. EU 고위험 의무 부담이 큽니다.
9. 시장 수치의 신뢰도가 낮습니다.
