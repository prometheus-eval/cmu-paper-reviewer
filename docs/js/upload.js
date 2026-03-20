const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const submitBtn = document.getElementById("submit-btn");
const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const modeInput = document.getElementById("mode");
const modePills = document.querySelectorAll(".mode-pill");

const codeInput = document.getElementById("code-input");
const suppInput = document.getElementById("supp-input");
const codeFileName = document.getElementById("code-file-name");
const suppFileName = document.getElementById("supp-file-name");

let selectedFile = null;

// ─── Review settings state ───────────────────────────────────────────────────

const NATURE_CRITERIA = [
  { name: "Validity", description: "Does the manuscript have significant flaws which should prohibit its publication?", importance: 1, enabled: true, custom: false },
  { name: "Conclusions", description: "Are the conclusions and data interpretation robust, valid and reliable?", importance: 2, enabled: true, custom: false },
  { name: "Originality and significance", description: "Are the results presented of immediate interest to many people in the field of study, and/or to people from several disciplines?", importance: 3, enabled: true, custom: false },
  { name: "Data and methodology", description: "Is the reporting of data and methodology sufficiently detailed and transparent to enable reproducing the results?", importance: 4, enabled: true, custom: false },
  { name: "Appropriate use of statistics and treatment of uncertainties", description: "Are all error bars defined in the corresponding figure legends and are all statistical tests appropriate and the description of any error bars and probability values accurate?", importance: 5, enabled: true, custom: false },
  { name: "Clarity and context", description: "Is the abstract clear, accessible? Are abstract, introduction and conclusions appropriate?", importance: 6, enabled: true, custom: false },
];

const NEURIPS_CRITERIA = [
  { name: "Originality", description: "Are the tasks or methods new? Is the work a novel combination of well-known techniques? Is it clear how this work differs from previous contributions? Is related work adequately cited?", importance: 1, enabled: true, custom: false },
  { name: "Quality", description: "Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?", importance: 2, enabled: true, custom: false },
  { name: "Clarity", description: "Is the submission clearly written? Is it well organized? Does it adequately inform the reader? (A superbly written paper provides enough information for an expert reader to reproduce its results.)", importance: 3, enabled: true, custom: false },
  { name: "Significance", description: "Are the results important? Are others likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way?", importance: 4, enabled: true, custom: false },
];

let reviewSettings = {
  enable_future_references: true,
  reviewer_criteria_preset: "nature",
  max_items: 5,
  criteria: JSON.parse(JSON.stringify(NATURE_CRITERIA)),
  criticize_limitations: true,
};

// --- Optional file inputs ---
codeInput.addEventListener("change", (e) => {
  codeFileName.textContent = e.target.files.length > 0 ? e.target.files[0].name : "";
});
suppInput.addEventListener("change", (e) => {
  suppFileName.textContent = e.target.files.length > 0 ? e.target.files[0].name : "";
});

// --- Pill switching ---
modePills.forEach((pill) => {
  pill.addEventListener("click", () => {
    const mode = pill.dataset.mode;
    modePills.forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    modeInput.value = mode;

    document.getElementById("queue-fields").style.display = mode === "queue" ? "" : "none";
    document.getElementById("queue-description").style.display = mode === "queue" ? "" : "none";
    document.getElementById("byok-fields").style.display = mode === "byok" ? "" : "none";
    document.getElementById("byok-description").style.display = mode === "byok" ? "" : "none";

    updateSubmitButton();
  });
});

// --- Drop zone ---
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    selectFile(e.target.files[0]);
  }
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    selectFile(e.dataTransfer.files[0]);
  }
});

function selectFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showMessage("error", "Please select a PDF file.");
    return;
  }
  selectedFile = file;
  dropZone.classList.add("has-file");
  const p = dropZone.querySelector("p");
  p.textContent = file.name;

  // Add clear button if not already present
  if (!dropZone.querySelector(".clear-file")) {
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "clear-file";
    clearBtn.textContent = "Remove";
    clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile();
    });
    dropZone.appendChild(clearBtn);
  }
  updateSubmitButton();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  dropZone.classList.remove("has-file");
  dropZone.querySelector("p").innerHTML =
    'Drag & drop a PDF here, or <strong>click to browse</strong>';
  const clearBtn = dropZone.querySelector(".clear-file");
  if (clearBtn) clearBtn.remove();
  updateSubmitButton();
}

function showMessage(type, html) {
  resultDiv.style.display = "block";
  resultDiv.className = "message " + type;
  resultDiv.innerHTML = html;
}

// --- Copy to clipboard ---
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector(".copy-btn");
    if (btn) {
      const orig = btn.innerHTML;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
      setTimeout(() => { btn.innerHTML = orig; }, 1500);
    }
  });
}

// --- Validation ---
function updateSubmitButton() {
  const mode = modeInput.value;

  if (!selectedFile) {
    submitBtn.disabled = true;
    return;
  }

  if (mode === "queue") {
    const email = document.getElementById("email").value.trim();
    submitBtn.disabled = !email;
  } else {
    const mistral = document.getElementById("mistral-key").value.trim();
    const litellm = document.getElementById("litellm-key").value.trim();
    submitBtn.disabled = !mistral || !litellm;
  }
}

// Listen for input changes on relevant fields
["email", "mistral-key", "litellm-key"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("input", updateSubmitButton);
});

// ─── Settings Panel ──────────────────────────────────────────────────────────

function toggleSettingsPanel() {
  const panel = document.getElementById("settings-panel");
  const chevron = document.querySelector(".settings-chevron");
  const isOpen = panel.style.display !== "none";
  panel.style.display = isOpen ? "none" : "block";
  if (chevron) chevron.style.transform = isOpen ? "" : "rotate(180deg)";
}

// Max items slider
const maxItemsSlider = document.getElementById("setting-max-items");
const maxItemsValue = document.getElementById("max-items-value");
maxItemsSlider.addEventListener("input", () => {
  maxItemsValue.textContent = maxItemsSlider.value;
  reviewSettings.max_items = parseInt(maxItemsSlider.value, 10);
});

// Future references toggle
document.getElementById("setting-future-refs").addEventListener("change", (e) => {
  reviewSettings.enable_future_references = e.target.checked;
});

// Criticize limitations toggle
document.getElementById("setting-criticize-limitations").addEventListener("change", (e) => {
  reviewSettings.criticize_limitations = e.target.checked;
});

// Criteria preset selector
const presetSelect = document.getElementById("setting-criteria-preset");
presetSelect.addEventListener("change", () => {
  const preset = presetSelect.value;
  reviewSettings.reviewer_criteria_preset = preset;
  if (preset === "nature") {
    reviewSettings.criteria = JSON.parse(JSON.stringify(NATURE_CRITERIA));
  } else if (preset === "neurips") {
    reviewSettings.criteria = JSON.parse(JSON.stringify(NEURIPS_CRITERIA));
  }
  // "custom" keeps current criteria
});

// Configure criteria button → opens modal
document.getElementById("configure-criteria-btn").addEventListener("click", () => {
  openCriteriaModal();
});

// ─── Criteria Configuration Modal ────────────────────────────────────────────

function openCriteriaModal() {
  const modalRoot = document.getElementById("criteria-modal-root");
  const criteria = reviewSettings.criteria;

  let criteriaHtml = "";
  criteria.sort((a, b) => a.importance - b.importance);
  for (let i = 0; i < criteria.length; i++) {
    const c = criteria[i];
    criteriaHtml += `
      <div class="criteria-config-item" data-index="${i}">
        <div class="criteria-config-row">
          <label class="toggle-switch toggle-sm">
            <input type="checkbox" class="criteria-enable" data-index="${i}" ${c.enabled ? "checked" : ""}>
            <span class="toggle-slider"></span>
          </label>
          <div class="criteria-config-info">
            <div class="criteria-config-name">${escapeHtml(c.name)}</div>
            <div class="criteria-config-desc">${escapeHtml(c.description || "")}</div>
          </div>
          <div class="criteria-config-importance">
            <label class="criteria-importance-label">Priority</label>
            <input type="number" class="criteria-importance-input" data-index="${i}" value="${c.importance}" min="1" max="20">
          </div>
          ${c.custom ? `<button class="criteria-remove-btn" data-index="${i}" title="Remove">&times;</button>` : ""}
        </div>
      </div>`;
  }

  modalRoot.innerHTML = `
    <div class="modal-overlay" id="criteria-overlay">
      <div class="modal" style="max-width:800px;">
        <h3>Configure Evaluation Criteria</h3>
        <p class="modal-subtitle">Enable/disable criteria, adjust their priority (lower number = higher priority), or add custom criteria.</p>

        <div class="criteria-presets-row">
          <button class="btn btn-secondary btn-sm" id="modal-load-nature">Load Nature Criteria</button>
          <button class="btn btn-secondary btn-sm" id="modal-load-neurips">Load NeurIPS Criteria</button>
        </div>

        <div id="criteria-list" class="criteria-list">
          ${criteriaHtml}
        </div>

        <div class="add-criteria-section">
          <div class="add-criteria-header">Add Custom Criterion</div>
          <div class="add-criteria-fields">
            <input type="text" id="new-criteria-name" placeholder="Criterion name" class="add-criteria-input">
            <input type="text" id="new-criteria-desc" placeholder="Description (optional)" class="add-criteria-input" style="flex:2;">
            <button class="btn btn-secondary btn-sm" id="add-criteria-btn" style="margin-top:0;">Add</button>
          </div>
        </div>

        <div class="modal-actions">
          <button class="btn btn-secondary btn-sm" id="criteria-cancel">Cancel</button>
          <button class="btn btn-primary btn-sm" id="criteria-save" style="margin-top:0;">Save</button>
        </div>
      </div>
    </div>`;

  // Wire up events
  document.getElementById("criteria-overlay").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeCriteriaModal();
  });
  document.getElementById("criteria-cancel").addEventListener("click", closeCriteriaModal);

  document.getElementById("modal-load-nature").addEventListener("click", () => {
    reviewSettings.criteria = JSON.parse(JSON.stringify(NATURE_CRITERIA));
    presetSelect.value = "nature";
    reviewSettings.reviewer_criteria_preset = "nature";
    openCriteriaModal(); // re-render
  });
  document.getElementById("modal-load-neurips").addEventListener("click", () => {
    reviewSettings.criteria = JSON.parse(JSON.stringify(NEURIPS_CRITERIA));
    presetSelect.value = "neurips";
    reviewSettings.reviewer_criteria_preset = "neurips";
    openCriteriaModal(); // re-render
  });

  // Enable/disable checkboxes
  document.querySelectorAll(".criteria-enable").forEach((cb) => {
    cb.addEventListener("change", () => {
      const idx = parseInt(cb.dataset.index, 10);
      reviewSettings.criteria[idx].enabled = cb.checked;
    });
  });

  // Importance inputs
  document.querySelectorAll(".criteria-importance-input").forEach((inp) => {
    inp.addEventListener("change", () => {
      const idx = parseInt(inp.dataset.index, 10);
      reviewSettings.criteria[idx].importance = parseInt(inp.value, 10) || 1;
    });
  });

  // Remove buttons
  document.querySelectorAll(".criteria-remove-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.index, 10);
      reviewSettings.criteria.splice(idx, 1);
      presetSelect.value = "custom";
      reviewSettings.reviewer_criteria_preset = "custom";
      openCriteriaModal(); // re-render
    });
  });

  // Add custom criterion
  document.getElementById("add-criteria-btn").addEventListener("click", () => {
    const name = document.getElementById("new-criteria-name").value.trim();
    const desc = document.getElementById("new-criteria-desc").value.trim();
    if (!name) return;
    const maxImp = Math.max(...reviewSettings.criteria.map((c) => c.importance), 0);
    reviewSettings.criteria.push({
      name,
      description: desc,
      importance: maxImp + 1,
      enabled: true,
      custom: true,
    });
    presetSelect.value = "custom";
    reviewSettings.reviewer_criteria_preset = "custom";
    openCriteriaModal(); // re-render
  });

  // Save
  document.getElementById("criteria-save").addEventListener("click", () => {
    closeCriteriaModal();
  });
}

function closeCriteriaModal() {
  document.getElementById("criteria-modal-root").innerHTML = "";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- Form submission ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  const mode = modeInput.value;
  submitBtn.disabled = true;
  submitBtn.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("file", selectedFile);

  if (mode === "queue") {
    formData.append("email", document.getElementById("email").value);
  } else {
    // BYOK mode
    const byokEmail = document.getElementById("byok-email").value.trim();
    if (byokEmail) formData.append("email", byokEmail);
    formData.append("user_mistral_api_key", document.getElementById("mistral-key").value);
    formData.append("user_litellm_api_key", document.getElementById("litellm-key").value);
    const litellmUrl = document.getElementById("litellm-url").value.trim();
    if (litellmUrl) formData.append("user_litellm_base_url", litellmUrl);
    const tavily = document.getElementById("tavily-key").value.trim();
    if (tavily) formData.append("user_tavily_api_key", tavily);
  }

  // Optional files
  if (codeInput.files.length > 0) {
    formData.append("code_file", codeInput.files[0]);
  }
  if (suppInput.files.length > 0) {
    formData.append("supplementary_file", suppInput.files[0]);
  }

  // Review settings
  formData.append("review_settings", JSON.stringify(reviewSettings));

  try {
    const resp = await fetch(`${API_BASE_URL}/api/submit`, {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Submission failed.");
    }

    const data = await resp.json();

    const copyIcon = `<button class="copy-btn" onclick="copyToClipboard('${data.key}')" title="Copy key"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>`;

    let successMsg;
    if (data.mode === "byok") {
      successMsg =
        `<strong>Submitted!</strong> Your paper is queued for <strong>priority processing</strong> using your API keys.<br>` +
        `Your submission key is:
         <div class="key-display"><span>${data.key}</span>${copyIcon}</div>
         <p>Use this key to retrieve your review on the
         <a href="review.html?key=${data.key}">review page</a>.
         Your API keys will be cleared after processing.</p>`;
    } else {
      successMsg =
        `<strong>Submitted!</strong> Your submission key is:
         <div class="key-display"><span>${data.key}</span>${copyIcon}</div>
         <p>We'll email you when the review is ready. You can also check status on the
         <a href="review.html?key=${data.key}">review page</a>.</p>`;
    }

    showMessage("success", successMsg);
  } catch (err) {
    if (err.message === "Failed to fetch") {
      showMessage("error",
        `<strong>Error:</strong> Could not reach the server at <code>${API_BASE_URL}</code>. ` +
        `Please check that the backend is running and CORS is configured for this origin.`);
    } else {
      showMessage("error", `<strong>Error:</strong> ${err.message}`);
    }
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Paper";
  }
});
