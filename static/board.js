"use strict";

const boardToken = consumeToken("board");
const authSection = document.querySelector("#board-auth");
const authForm = document.querySelector("#board-auth-form");
const authInput = document.querySelector("#board-token");
const authStatus = document.querySelector("#auth-status");
const board = document.querySelector("#board-content");
const boardStatus = document.querySelector("#board-status");
const connection = document.querySelector("#connection");
const connectionLabel = document.querySelector("#connection-label");
const pendingQueue = document.querySelector("#pending-queue");
const pendingList = document.querySelector("#pending-cards");
const completedQueue = document.querySelector("#completed-queue");
const queue = document.querySelector("#cards");
const template = document.querySelector("#card-template");
const outageRecoveryToggle = document.querySelector("#outage-recovery");
const outageRecoveryStatus = document.querySelector("#recovery-status");
const ruleForm = document.querySelector("#rule-form");
const ruleInput = document.querySelector("#rule-text");
const burstButton = document.querySelector("#run-burst");
const progressBox = document.querySelector("#burst-progress");
const burstLabel = document.querySelector("#burst-label");
const burstMeter = document.querySelector("#burst-meter");
const burstElapsed = document.querySelector("#burst-elapsed");
const burstRate = document.querySelector("#burst-rate");
const burstInterval = document.querySelector("#burst-interval");

const urgencyNames = {
  top: "최우선",
  urgent: "긴급",
  normal: "일반",
  non_urgent: "비긴급",
};
const probabilityNames = {
  people: "위험 인원",
  now: "글의 진행 여부",
  severe: "심각한 사진 피해",
};
const optionalProbabilityNames = {
  life_threat: "즉각적인 생명 위협",
  ongoing_hazard: "사진 속 진행 위험",
  field_photo: "현장 사진",
};
const personStateNames = {
  life_threat: "즉각적인 생명 위협",
  trapped: "고립",
  minor_injury: "경상",
  none: "인명 위험 없음",
};
const personPercent = new Intl.NumberFormat("ko-KR", { style: "percent", maximumFractionDigits: 2 });
const disasterNames = {
  fire: "화재",
  storm_flood: "폭풍·홍수",
  quake_collapse: "지진·구조물 붕괴",
  other: "기타·유형 불명",
  fire_smoke_burn: "불꽃·연기·소실",
  water_storm: "물·바람 피해",
  collapse: "구조물 붕괴",
  unclear: "판단 불가",
};
const modeNames = {
  zoro: "Zoro 분석",
  fallback: "폴백 판정",
};
const recoveryNames = {
  scheduled: "재판정 대기",
  waiting_for_outage: "복구 대기",
  retrying: "재판정 중",
  recovered: "재판정 완료",
  failed: "재판정 실패",
};
const dateFormat = new Intl.DateTimeFormat("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const cardNodes = new Map();
let stream;
let pollInterval;
let lastVersion;
let rulesSignature = "";
let activeBurstRequest = false;
let burstRequestId;
const exampleRequestIds = new Map();
let burstRunning = false;
let burstState;
let burstStartedTick;
let burstTimeInterval;
let outageRecoveryEnabled;
let activeRecoveryRequest = false;
let recoveryTimeInterval;
let activeRuleRequest = false;
let ruleCount = 0;

function setConnection(state, text) {
  connection.dataset.state = state;
  connectionLabel.textContent = text;
}

function stopUpdates() {
  if (stream) stream.close();
  stream = undefined;
  clearInterval(pollInterval);
  pollInterval = undefined;
}

function showAuth(message = "") {
  stopUpdates();
  stopBurstClock();
  stopRecoveryClock();
  lastVersion = undefined;
  authSection.hidden = false;
  board.hidden = true;
  if (message) {
    boardToken.value = "";
    boardToken.save("");
  }
  setStatus(authStatus, message, message ? "error" : "");
  setConnection("offline", "연결 대기");
}

function showConnectionError(error) {
  authSection.hidden = false;
  board.hidden = true;
  setStatus(authStatus, error.message, "error");
  setConnection("offline", "연결 오류");
}

function openBoard(state) {
  authSection.hidden = true;
  board.hidden = false;
  renderState(state);
  subscribe();
}

async function login(token) {
  await request("/api/session", { data: { token } });
  boardToken.value = token;
  boardToken.save(token);
  const state = await request("/api/state");
  authInput.value = "";
  openBoard(state);
}

async function initialize() {
  try {
    if (boardToken.value) {
      await login(boardToken.value);
    } else {
      openBoard(await request("/api/state"));
    }
  } catch (error) {
    if (needsToken(error)) showAuth(boardToken.value ? error.message : "");
    else showConnectionError(error);
  }
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = authForm.querySelector("button");
  const token = authInput.value.trim();
  if (!token || button.disabled) return;
  button.disabled = true;
  setStatus(authStatus, "상황판에 연결하고 있습니다.");
  try {
    await login(token);
  } catch (error) {
    if (needsToken(error)) showAuth(error.message);
    else showConnectionError(error);
  } finally {
    button.disabled = false;
  }
});

let pollRequest = false;
function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(poll, 2000);
  void poll();
}

async function poll() {
  if (pollRequest) return;
  const activeInterval = pollInterval;
  pollRequest = true;
  try {
    const state = await request("/api/state");
    if (activeInterval === pollInterval && !board.hidden) {
      renderState(state);
      setConnection("polling", "자동 갱신 중");
    }
  } catch (error) {
    if (activeInterval === pollInterval) {
      if (needsToken(error)) showAuth(error.message);
      else setConnection("offline", "재연결 중");
    }
  } finally {
    pollRequest = false;
  }
}

function subscribe() {
  stopUpdates();
  if (!window.EventSource) {
    startPolling();
    return;
  }
  setConnection("polling", "실시간 연결 중");
  stream = new EventSource("/api/events");
  stream.addEventListener("state", (event) => {
    let state;
    try {
      state = JSON.parse(event.data);
    } catch {
      startPolling();
      return;
    }
    clearInterval(pollInterval);
    pollInterval = undefined;
    setConnection("live", "실시간 연결");
    renderState(state);
  });
  stream.addEventListener("error", () => {
    setConnection("offline", "재연결 중");
    startPolling();
  });
}

function prettyTime(milliseconds) {
  if (milliseconds < 1000) return `${Math.round(milliseconds)}ms`;
  return `${(milliseconds / 1000).toFixed(1)}초`;
}

function syncBurstButton() {
  burstButton.disabled = activeBurstRequest || burstRunning;
}

function updateBurstLabel() {
  const { total, done, elapsed_ms } = burstState;
  const elapsed = burstRunning ? performance.now() - burstStartedTick : elapsed_ms;
  const label = `${done}/${total}건 완료`;
  if (burstLabel.textContent !== label) burstLabel.textContent = label;
  burstElapsed.textContent = Number.isFinite(elapsed) ? prettyTime(elapsed) : "—";
  burstRate.textContent = elapsed > 0 ? `${(done * 1000 / elapsed).toFixed(2)}건/초` : "—";
  burstInterval.textContent = done > 0 && Number.isFinite(elapsed) ? prettyTime(elapsed / done) : "—";
}

function stopBurstClock() {
  clearInterval(burstTimeInterval);
  burstTimeInterval = undefined;
}

function renderBurst(burst) {
  const running = Boolean(burst && burst.done < burst.total);
  if (running) {
    const now = performance.now();
    if (!burstRunning || burst.id !== burstState.id) {
      const elapsed = Number.isFinite(burst.elapsed_ms) ? burst.elapsed_ms : 0;
      burstStartedTick = now - Math.max(0, elapsed);
    }
    if (Number.isFinite(burst.elapsed_ms)) {
      burstStartedTick = Math.min(burstStartedTick, now - burst.elapsed_ms);
    }
    burstTimeInterval ??= setInterval(updateBurstLabel, 100);
  } else {
    stopBurstClock();
  }
  burstState = burst;
  burstRunning = running;
  progressBox.hidden = !burst;
  if (burst) {
    burstMeter.max = burst.total;
    burstMeter.value = burst.done;
    updateBurstLabel();
  }
  syncBurstButton();
}

function syncRuleButton() {
  ruleForm.querySelector("button").disabled = activeRuleRequest || ruleCount >= 8;
  ruleInput.disabled = ruleCount >= 8;
  ruleInput.placeholder = ruleCount >= 8 ? "규칙 8개 등록 완료" : "고립된 고령자·어린이는 최우선";
}

function syncOutageRecoveryToggle() {
  outageRecoveryToggle.disabled = activeRecoveryRequest || typeof outageRecoveryEnabled !== "boolean";
  if (!activeRecoveryRequest) outageRecoveryToggle.checked = outageRecoveryEnabled === true;
}

function stopRecoveryClock() {
  clearInterval(recoveryTimeInterval);
  recoveryTimeInterval = undefined;
}

function updateRecoveryClock() {
  const now = Date.now() / 1000;
  let countingDown = false;
  for (const view of cardNodes.values()) {
    if (!Number.isFinite(view.recoveryDueAt)) continue;
    const seconds = Math.ceil(view.recoveryDueAt - now);
    const label = seconds > 0 ? `${recoveryNames.scheduled} · ${seconds}초` : recoveryNames.scheduled;
    if (view.recovery.textContent !== label) view.recovery.textContent = label;
    if (seconds > 0) countingDown = true;
    else view.recoveryDueAt = undefined;
  }
  if (countingDown) recoveryTimeInterval ??= setInterval(updateRecoveryClock, 250);
  else stopRecoveryClock();
}

function renderState(state) {
  if (!Array.isArray(state.cards) || !Array.isArray(state.rules)) return;
  if (Number.isFinite(state.version) && state.version === lastVersion) return;
  lastVersion = state.version;
  const cards = state.cards;
  const pending = [];
  const completed = [];
  for (const card of cards) {
    (card.status === "pending" ? pending : completed).push(card);
  }
  document.querySelector("#total-count").textContent = cards.length;
  document.querySelector("#pending-count").textContent = pending.length;
  document.querySelector("#top-count").textContent = completed.filter((card) => (card.human_urgency || card.urgency) === "top").length;
  document.querySelector("#label-count").textContent = state.labels;
  document.querySelector("#empty-queue").hidden = cards.length > 0;
  pendingQueue.hidden = pending.length === 0;
  completedQueue.hidden = completed.length === 0;
  document.querySelector("#rule-revision").textContent = `v${state.revision}`;
  const nextRulesSignature = JSON.stringify(state.rules);
  if (nextRulesSignature !== rulesSignature) {
    rulesSignature = nextRulesSignature;
    const items = state.rules.map((rule) => {
      const item = document.createElement("li");
      item.textContent = rule;
      return item;
    });
    document.querySelector("#rules").replaceChildren(...items);
    ruleCount = items.length;
    syncRuleButton();
  }
  outageRecoveryEnabled = state.outage_recovery;
  syncOutageRecoveryToggle();
  renderBurst(state.burst);
  removeMissingCards(cards);
  renderCards(pendingList, pending);
  renderCards(queue, completed);
  updateRecoveryClock();
}

function removeMissingCards(cards) {
  const currentIds = new Set(cards.map((card) => card.id));
  for (const [id, view] of cardNodes) {
    if (!currentIds.has(id)) {
      view.element.remove();
      cardNodes.delete(id);
    }
  }
}

function renderCards(queue, cards) {
  let cursor = queue.firstElementChild;
  for (const card of cards) {
    if (!cardNodes.has(card.id)) cardNodes.set(card.id, createCard(card.id));
    const view = cardNodes.get(card.id);
    updateCard(view, card);
    if (view.element === cursor) {
      cursor = cursor.nextElementSibling;
    } else if (typeof queue.moveBefore === "function" && view.element.parentNode === queue) {
      queue.moveBefore(view.element, cursor);
    } else {
      const active = document.activeElement;
      const hadFocus = view.element.contains(active);
      queue.insertBefore(view.element, cursor);
      if (hadFocus) active.focus({ preventScroll: true });
    }
  }
}

function createCard(id) {
  const element = template.content.firstElementChild.cloneNode(true);
  const labelForm = element.querySelector(".label-form");
  const review = element.querySelector(".review-flag");
  const fieldPhotoReview = document.createElement("span");
  fieldPhotoReview.className = "review-flag";
  fieldPhotoReview.textContent = "확인 필요(현장 사진 아님)";
  fieldPhotoReview.hidden = true;
  review.after(fieldPhotoReview);
  const view = {
    element,
    image: element.querySelector("img"),
    time: element.querySelector("time"),
    text: element.querySelector(".card-text"),
    urgency: element.querySelector(".urgency"),
    source: element.querySelector(".card-source"),
    recovery: element.querySelector(".recovery-flag"),
    review,
    fieldPhotoReview,
    details: element.querySelector(".card-details"),
    reason: element.querySelector(".ai-reason"),
    meta: element.querySelector(".decision-meta"),
    probabilities: element.querySelector(".probabilities"),
    field: labelForm.elements.urgency,
    status: element.querySelector(".label-status"),
    signature: "",
  };
  labelForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = labelForm.querySelector("button");
    if (!view.field.value || button.disabled) return;
    button.disabled = true;
    view.field.disabled = true;
    try {
      await request(`/api/reports/${encodeURIComponent(id)}/label`, {
        data: { urgency: view.field.value, revision: view.revision },
      });
      setStatus(view.status, "사람 검수를 저장했습니다.", "success");
    } catch (error) {
      if (needsToken(error)) showAuth(error.message);
      else setStatus(view.status, error.message, "error");
    } finally {
      view.field.disabled = false;
      button.disabled = false;
    }
  });
  return view;
}

function updateCard(view, card) {
  const signature = JSON.stringify(card);
  if (view.signature === signature) return;
  view.signature = signature;
  view.revision = card.rule_revision;

  const decision = card.decision;
  const classified = card.status === "done";
  const urgency = card.human_urgency || card.urgency || decision?.urgency || "";
  view.element.dataset.urgency = classified ? urgency : "";
  view.urgency.textContent = classified ? (urgencyNames[urgency] || "분류 완료") : "분석 대기";
  const sourceLabel = { example: "훈련 예시", burst: "폭주 시뮬레이션", phone: "현장 제보" }[card.source];
  view.source.hidden = !sourceLabel;
  view.source.textContent = classified && decision?.mode === "fallback" ? "폴백 판정" : sourceLabel || "";
  const recoveryStatus = classified ? card.recovery?.status : undefined;
  const recoveryLabel = recoveryNames[recoveryStatus] || "";
  view.recovery.hidden = !recoveryLabel;
  view.recovery.dataset.state = recoveryStatus || "";
  view.recovery.textContent = recoveryLabel;
  view.recoveryDueAt = recoveryStatus === "scheduled" ? card.recovery.due_at : undefined;
  view.text.textContent = card.text;
  if (card.image_url && view.image.getAttribute("src") !== card.image_url) view.image.src = card.image_url;
  const date = new Date(card.created_at * 1000);
  if (Number.isFinite(date.getTime())) {
    view.time.dateTime = date.toISOString();
    view.time.textContent = dateFormat.format(date);
  }
  view.details.hidden = !classified;
  const nonFieldPhoto = decision?.non_field_photo === true;
  view.fieldPhotoReview.hidden = !classified || !nonFieldPhoto;
  const uncertainGrade = Boolean(decision?.review_uncertainty?.length);
  view.review.hidden = !classified || !(card.human_urgency || (decision?.review && (!nonFieldPhoto || uncertainGrade)));
  view.review.textContent = card.human_urgency ? "사람 검수 완료"
    : uncertainGrade ? "확인 필요(판단 불확실)" : "사람 확인 필요";
  view.review.dataset.done = String(Boolean(card.human_urgency));
  const reasons = classified && Array.isArray(decision?.reasons) ? decision.reasons : [];
  view.reason.hidden = !reasons.length;
  view.reason.textContent = reasons.join(" · ");
  const details = [];
  if (decision?.mode) details.push(modeNames[decision.mode] || decision.mode);
  if (decision?.urgency && urgencyNames[decision.urgency]) {
    const classificationLabel = recoveryStatus === "recovered" ? "재판정 분류" : "초기 분류";
    details.push(`${classificationLabel} ${urgencyNames[decision.urgency]}`);
  }
  if (Number.isInteger(card.rule_revision)) details.push(`규칙 v${card.rule_revision}`);
  view.meta.textContent = details.join(" · ");
  if (card.human_urgency && card.human_urgency !== view.savedLabel) view.field.value = card.human_urgency;
  view.savedLabel = card.human_urgency;
  const probabilities = [];
  if (classified && decision) {
    for (const [key, label] of Object.entries(probabilityNames)) {
      const value = decision.probs?.[key];
      probabilities.push(metric(label, percent(value)));
    }
    const personProbabilities = decision.person_probabilities;
    if (personProbabilities) {
      probabilities.push(metric("인명 상태", personStateNames[decision.person_state]));
      for (const [key, label] of Object.entries(personStateNames)) {
        probabilities.push(metric(label, personPercent.format(personProbabilities[key])));
      }
      probabilities.push(metric("생명 위협 판정 기준", personPercent.format(decision.person_threshold)));
    }
    for (const [key, label] of Object.entries(optionalProbabilityNames)) {
      if (key === "life_threat" && personProbabilities) continue;
      const value = decision.probs?.[key];
      if (value !== undefined) probabilities.push(metric(label, percent(value)));
    }
    if (typeof decision.mismatch === "boolean") {
      probabilities.push(
        metric("신고 글 유형", disasterType(decision.claim_kind, decision.claim_prob)),
        metric("사진 유형", disasterType(decision.photo_kind, decision.photo_prob)),
        metric("사진·설명 불일치", decision.mismatch ? "감지" : "미감지"),
      );
    } else {
      probabilities.push(metric("사진·설명 불일치", percent(decision.probs?.mismatch)));
    }
    if (Number.isFinite(decision.ms)) {
      probabilities.push(metric("판정 지연", `${Math.round(decision.ms)} ms`));
    }
    if (Number.isFinite(card.first_decision_with_wait_ms)) {
      probabilities.push(metric("접수→최초 판정 (대기 포함)", prettyTime(card.first_decision_with_wait_ms)));
    }
  }
  view.probabilities.replaceChildren(...probabilities);
}

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "미측정";
}

function disasterType(kind, confidence) {
  if (!kind) return "미측정";
  return `${disasterNames[kind] || kind} (${percent(confidence)})`;
}

function metric(label, value) {
  const group = document.createElement("div");
  const term = document.createElement("dt");
  const number = document.createElement("dd");
  term.textContent = label;
  number.textContent = value;
  group.append(term, number);
  return group;
}

function handleBoardError(error) {
  if (needsToken(error)) showAuth(error.message);
  else setStatus(boardStatus, error.message, "error");
}

outageRecoveryToggle.addEventListener("change", async () => {
  if (outageRecoveryToggle.disabled) return;
  const enabled = outageRecoveryToggle.checked;
  const requestedAtVersion = lastVersion;
  activeRecoveryRequest = true;
  syncOutageRecoveryToggle();
  try {
    const result = await request("/api/outage-recovery", { data: { enabled } });
    if (lastVersion === requestedAtVersion) outageRecoveryEnabled = result.enabled;
    setStatus(outageRecoveryStatus, "");
  } catch (error) {
    if (needsToken(error)) showAuth(error.message);
    else setStatus(outageRecoveryStatus, error.message, "error");
  } finally {
    activeRecoveryRequest = false;
    syncOutageRecoveryToggle();
  }
});

for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", async () => {
    if (button.disabled) return;
    const kind = button.dataset.example;
    const requestId = exampleRequestIds.get(kind) || createRequestId();
    exampleRequestIds.set(kind, requestId);
    button.disabled = true;
    setStatus(boardStatus, "훈련 예시를 접수하고 있습니다.");
    try {
      await request("/api/example", { data: { kind, request_id: requestId } });
      exampleRequestIds.delete(kind);
      setStatus(boardStatus, "훈련 예시를 접수했습니다.", "success");
    } catch (error) {
      if (!requestOutcomeUnknown(error)) exampleRequestIds.delete(kind);
      handleBoardError(error);
    } finally {
      button.disabled = false;
    }
  });
}

ruleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = ruleInput.value.trim();
  if (!text || activeRuleRequest || ruleCount >= 8) return;
  activeRuleRequest = true;
  syncRuleButton();
  try {
    await request("/api/rules", { data: { text } });
    if (ruleInput.value.trim() === text) ruleInput.value = "";
    setStatus(boardStatus, "우선순위 규칙을 추가했습니다.", "success");
  } catch (error) {
    handleBoardError(error);
  } finally {
    activeRuleRequest = false;
    syncRuleButton();
  }
});

burstButton.addEventListener("click", async () => {
  if (burstButton.disabled) return;
  burstRequestId ||= createRequestId();
  activeBurstRequest = true;
  syncBurstButton();
  setStatus(boardStatus, "제보 100건을 투입하고 있습니다.");
  try {
    await request("/api/burst", { data: { request_id: burstRequestId } });
    burstRequestId = undefined;
    setStatus(boardStatus, "제보 100건을 투입했습니다.", "success");
  } catch (error) {
    if (!requestOutcomeUnknown(error)) burstRequestId = undefined;
    handleBoardError(error);
  } finally {
    activeBurstRequest = false;
    syncBurstButton();
  }
});

void initialize();
