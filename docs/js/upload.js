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
