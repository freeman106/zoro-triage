"use strict";

const reportToken = consumeToken("report");
const form = document.querySelector("#report-form");
const authField = document.querySelector("#report-auth");
const tokenInput = document.querySelector("#report-token");
const reportText = document.querySelector("#report-text");
const camera = document.querySelector("#camera");
const gallery = document.querySelector("#gallery");
const preview = document.querySelector("#photo-preview");
const emptyPhoto = document.querySelector("#photo-empty");
const photoName = document.querySelector("#photo-name");
const submit = document.querySelector("#send-report");
const statusLine = document.querySelector("#report-status");

let imageData = "";
let selection = 0;
let pendingPhoto = false;
let submitted;

function setAuthVisible(visible) {
  authField.hidden = !visible;
  tokenInput.required = visible;
}

setAuthVisible(!reportToken.value);

function clearPhoto() {
  imageData = "";
  preview.removeAttribute("src");
  preview.hidden = true;
  photoName.hidden = true;
  emptyPhoto.hidden = false;
}

async function asJpeg(file) {
  const url = URL.createObjectURL(file);
  try {
    const original = new Image();
    original.src = url;
    await original.decode();
    const ratio = Math.min(1, 1280 / Math.max(original.naturalWidth, original.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(original.naturalWidth * ratio));
    canvas.height = Math.max(1, Math.round(original.naturalHeight * ratio));
    const context = canvas.getContext("2d", { alpha: false });
    context.fillStyle = "#fff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(original, 0, 0, canvas.width, canvas.height);
    const jpeg = canvas.toDataURL("image/jpeg", 0.8);
    if (!jpeg.startsWith("data:image/jpeg;base64,")) throw new Error("JPEG conversion failed");
    return jpeg;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function selectPhoto(event) {
  const input = event.currentTarget;
  const file = input.files[0];
  input.value = "";
  if (!file) return;
  const currentSelection = ++selection;
  pendingPhoto = true;
  submit.disabled = true;
  setStatus(statusLine, "사진을 준비하고 있습니다.");
  try {
    const jpeg = await asJpeg(file);
    if (currentSelection !== selection) return;
    imageData = jpeg.slice(jpeg.indexOf(",") + 1);
    preview.src = jpeg;
    preview.hidden = false;
    emptyPhoto.hidden = true;
    photoName.textContent = file.name;
    photoName.hidden = false;
    setStatus(statusLine, "");
  } catch {
    if (currentSelection !== selection) return;
    setStatus(statusLine, "이 사진을 읽을 수 없습니다. JPEG 또는 PNG 사진을 선택하세요.", "error");
  } finally {
    if (currentSelection === selection) {
      pendingPhoto = false;
      submit.disabled = false;
    }
  }
}

camera.addEventListener("change", selectPhoto);
gallery.addEventListener("change", selectPhoto);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (pendingPhoto || submit.disabled) return;
  const text = reportText.value.trim();
  if (!text) {
    setStatus(statusLine, "상황 설명을 입력해주세요.", "error");
    reportText.focus();
    return;
  }
  if (!imageData) {
    setStatus(statusLine, "현장 사진 한 장을 선택해주세요.", "error");
    camera.focus();
    return;
  }

  const token = authField.hidden ? reportToken.value : tokenInput.value.trim();
  if (!submitted || submitted.text !== text || submitted.image !== imageData) {
    submitted = { text, image: imageData, request_id: createRequestId() };
  }
  submit.disabled = true;
  reportText.disabled = true;
  camera.disabled = true;
  gallery.disabled = true;
  tokenInput.disabled = true;
  setStatus(statusLine, "상황실로 제보를 보내고 있습니다.");
  try {
    const result = await request("/api/reports", { data: submitted, token });
    submitted = undefined;
    reportToken.value = token;
    reportToken.save(token);
    form.reset();
    setAuthVisible(false);
    clearPhoto();
    const receipt = result.id ? ` · 접수 번호 ${result.id}` : "";
    setStatus(statusLine, `제보가 접수됐습니다.${receipt}`, "success");
  } catch (error) {
    if (!requestOutcomeUnknown(error)) submitted = undefined;
    if (needsToken(error)) {
      reportToken.value = "";
      reportToken.save("");
      setAuthVisible(true);
    }
    setStatus(statusLine, error.message, "error");
  } finally {
    submit.disabled = false;
    reportText.disabled = false;
    camera.disabled = false;
    gallery.disabled = false;
    tokenInput.disabled = false;
  }
});
