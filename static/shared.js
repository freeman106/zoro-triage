"use strict";

function consumeToken(role) {
  const storageKey = `two-drone-119-${role}-token`;
  const hash = new URLSearchParams(location.hash.slice(1));
  const supplied = hash.get("token");
  if (hash.has("token")) history.replaceState(null, "", location.pathname + location.search);

  function save(token) {
    try {
      if (token) sessionStorage.setItem(storageKey, token);
      else sessionStorage.removeItem(storageKey);
    } catch {
      // The current tab can still use the in-memory token when storage is blocked.
    }
  }

  if (supplied) {
    save(supplied);
    return { value: supplied, save };
  }
  try {
    return { value: sessionStorage.getItem(storageKey) || "", save };
  } catch {
    return { value: "", save };
  }
}

class RequestError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, { data, token } = {}) {
  const headers = {};
  if (data !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(path, {
      method: data === undefined ? "GET" : "POST",
      credentials: "same-origin",
      headers,
      body: data === undefined ? undefined : JSON.stringify(data),
    });
  } catch {
    throw new RequestError(0, "연결이 끊겼습니다. 잠시 후 다시 시도하세요.");
  }
  if (!response.ok) {
    const messages = {
      400: "입력 내용을 확인하고 다시 시도하세요.",
      401: "접근 키를 확인해주세요.",
      403: "접근 키를 확인해주세요.",
      409: "이미 진행 중이거나 등록된 작업입니다.",
      413: "사진 또는 설명이 너무 큽니다.",
      422: "입력 내용을 확인하고 다시 시도하세요.",
      429: "접수 요청이 많습니다. 잠시 후 다시 시도하세요.",
    };
    throw new RequestError(response.status, messages[response.status] || "요청을 처리하지 못했습니다. 잠시 후 다시 시도하세요.");
  }
  try {
    return await response.json();
  } catch {
    throw new RequestError(0, "응답을 확인할 수 없습니다. 잠시 후 다시 시도하세요.");
  }
}

function setStatus(element, message, state = "") {
  element.textContent = message;
  element.dataset.state = state;
}

function needsToken(error) {
  return error.status === 401 || error.status === 403;
}

function createRequestId() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function requestOutcomeUnknown(error) {
  return error.status === 0 || error.status >= 500;
}
