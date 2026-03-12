const API_BASE = "http://127.0.0.1:8888";

const state = {
  modelKey: null,
  modelSpecs: null,
};

function fmtRange(field) {
  const hasMin = typeof field.min === "number";
  const hasMax = typeof field.max === "number";
  if (hasMin && hasMax) return `${field.min} – ${field.max}`;
  if (hasMin) return `≥ ${field.min}`;
  if (hasMax) return `≤ ${field.max}`;
  return "n/a";
}

function clearResult() {
  const el = document.getElementById("result");
  el.style.display = "none";
  el.classList.remove("ok", "bad");
  el.innerText = "";
}

function showView(which) {
  document.getElementById("chooserView").style.display = which === "chooser" ? "block" : "none";
  document.getElementById("predictView").style.display = which === "predict" ? "block" : "none";
}

function goHome() {
  state.modelKey = null;
  clearResult();
  showView("chooser");
}

function validateField(field, value) {
  if (Number.isNaN(value)) return "Enter a number";
  if (typeof field.min === "number" && value < field.min) return `Expected ≥ ${field.min}`;
  if (typeof field.max === "number" && value > field.max) return `Expected ≤ ${field.max}`;
  return "";
}

function updateValidationSummary() {
  const spec = state.modelSpecs?.[state.modelKey];
  const summary = document.getElementById("validationSummary");
  if (!spec) {
    summary.innerText = "";
    return;
  }

  let invalid = 0;
  for (const field of spec.fields) {
    const el = document.getElementById(`f-${field.key}`);
    if (!el) continue;
    const v = parseFloat(el.value);
    const msg = validateField(field, v);
    if (msg) invalid++;
  }

  summary.innerText = invalid === 0
    ? "All fields look valid."
    : `${invalid} field(s) need attention (check ranges).`;
}

function prettyDesc(spec) {
  const task = spec.task ?? "classification";
  return task === "regression" ? "Numeric prediction." : "Label prediction.";
}

function renderCards() {
  const holder = document.getElementById("modelCards");
  const count = document.getElementById("modelCount");
  holder.innerHTML = "";

  const keys = Object.keys(state.modelSpecs ?? {});
  console.debug("Loaded model keys:", keys);

  count.innerText = keys.length ? `${keys.length} models available` : "";

  // Prefer a stable order if these exist
  const preferredOrder = ["knn", "logistic", "svm", "mlr", "polynomial"];
  const ordered = [
    ...preferredOrder.filter((k) => keys.includes(k)),
    ...keys.filter((k) => !preferredOrder.includes(k)),
  ];

  // If you have more than 5 models in the future, still keep the landing page to 5 buttons
  const display = ordered.slice(0, 5);

  const colors = ["color-1", "color-2", "color-3", "color-4", "color-5"];

  display.forEach((k, idx) => {
    const spec = state.modelSpecs[k];

    const btn = document.createElement("button");
    btn.className = `bigBtn ${colors[idx % colors.length]}`;
    btn.type = "button";
    btn.addEventListener("click", () => selectModel(k));

    const title = document.createElement("div");
    title.className = "title";
    title.innerText = spec.title ?? k;

    const meta = document.createElement("div");
    meta.className = "meta";

    const t1 = document.createElement("span");
    t1.className = "tag";
    t1.innerText = `${spec.task ?? "classification"}`;

    const t2 = document.createElement("span");
    t2.className = "tag";
    t2.innerText = `${spec.fields?.length ?? 0} inputs`;

    meta.appendChild(t1);
    meta.appendChild(t2);

    const desc = document.createElement("div");
    desc.className = "desc";
    desc.innerText = prettyDesc(spec);

    btn.appendChild(title);
    btn.appendChild(meta);
    btn.appendChild(desc);

    holder.appendChild(btn);
  });
}

function renderForm() {
  const spec = state.modelSpecs?.[state.modelKey];
  const form = document.getElementById("form");
  const title = document.getElementById("predictTitle");
  const hint = document.getElementById("formHint");
  const endpoint = document.getElementById("endpointPill");

  form.innerHTML = "";
  clearResult();

  if (!spec) {
    title.innerText = "Inputs";
    hint.innerText = "";
    endpoint.style.display = "none";
    return;
  }

  title.innerText = spec.title;
  hint.innerText = "Enter values (ranges shown).";

  endpoint.style.display = "inline-flex";
  endpoint.innerText = `POST ${spec.endpoint}`;

  for (const field of spec.fields) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const top = document.createElement("div");
    top.className = "field-top";

    const label = document.createElement("label");
    label.setAttribute("for", `f-${field.key}`);
    label.innerText = field.label ?? field.key;

    const range = document.createElement("div");
    range.className = "range";
    range.innerText = `Range: ${fmtRange(field)}`;

    top.appendChild(label);
    top.appendChild(range);

    const input = document.createElement("input");
    input.id = `f-${field.key}`;
    input.placeholder = typeof field.example === "number" ? `e.g. ${field.example}` : (field.label ?? field.key);
    input.inputMode = "decimal";
    input.autocomplete = "off";
    input.step = "any";
    if (typeof field.min === "number") input.min = String(field.min);
    if (typeof field.max === "number") input.max = String(field.max);

    const metaRow = document.createElement("div");
    metaRow.className = "field-meta";

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerText = typeof field.example === "number" ? `Example: ${field.example}` : "";

    const err = document.createElement("div");
    err.className = "error";
    err.id = `e-${field.key}`;

    metaRow.appendChild(meta);
    metaRow.appendChild(err);

    input.addEventListener("input", () => {
      const v = parseFloat(input.value);
      const msg = validateField(field, v);
      err.innerText = msg;
      input.classList.toggle("invalid", Boolean(msg));
      updateValidationSummary();
    });

    wrapper.appendChild(top);
    wrapper.appendChild(input);
    wrapper.appendChild(metaRow);
    form.appendChild(wrapper);
  }

  updateValidationSummary();
}

function selectModel(k) {
  state.modelKey = k;
  renderForm();
  showView("predict");
}

function fillExamples() {
  const spec = state.modelSpecs?.[state.modelKey];
  if (!spec) return;
  for (const field of spec.fields) {
    const el = document.getElementById(`f-${field.key}`);
    if (!el) continue;
    if (typeof field.example === "number") {
      el.value = field.example;
      el.dispatchEvent(new Event("input"));
    }
  }
}

function collectData() {
  const spec = state.modelSpecs?.[state.modelKey];
  if (!spec) throw new Error("Model spec not loaded");

  const data = {};
  let invalidCount = 0;

  for (const field of spec.fields) {
    const el = document.getElementById(`f-${field.key}`);
    const errEl = document.getElementById(`e-${field.key}`);

    const v = parseFloat(el.value);
    const msg = validateField(field, v);

    errEl.innerText = msg;
    el.classList.toggle("invalid", Boolean(msg));

    if (msg) invalidCount++;
    data[field.key] = v;
  }

  updateValidationSummary();

  if (invalidCount > 0) throw new Error(`Fix ${invalidCount} field(s) before predicting.`);

  return data;
}

async function predict() {
  const spec = state.modelSpecs?.[state.modelKey];
  const out = document.getElementById("result");
  const btn = document.getElementById("predictBtn");

  try {
    if (!spec) throw new Error("Select a model first");
    clearResult();

    const data = collectData();

    btn.disabled = true;
    btn.innerText = "Predicting…";

    const res = await fetch(`${API_BASE}${spec.endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const payload = await res.json();
    if (!res.ok) throw new Error(payload?.detail ?? `HTTP ${res.status}`);

    out.style.display = "block";
    out.classList.remove("bad");
    out.classList.add("ok");

    const label = payload.label ? ` (${payload.label})` : "";
    out.innerText = `Prediction: ${payload.prediction}${label}`;
  } catch (e) {
    out.style.display = "block";
    out.classList.remove("ok");
    out.classList.add("bad");
    out.innerText = `Error: ${e.message ?? e}`;
  } finally {
    btn.disabled = false;
    btn.innerText = "Predict";
  }
}

async function loadModels() {
  const status = document.getElementById("status");

  try {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.modelSpecs = await res.json();

    status.innerText = "";
    renderCards();
    showView("chooser");
  } catch (e) {
    status.innerText = `Failed to load model definitions from backend: ${e}`;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("apiBaseText").innerText = API_BASE;

  // Wire global handlers for inline HTML attrs
  window.goHome = goHome;
  window.fillExamples = fillExamples;
  window.predict = predict;
  // Alias used by the static fallback buttons in index.html
  window.selectModelById = selectModel;
  window.selectModel = selectModel;

  loadModels();
});
