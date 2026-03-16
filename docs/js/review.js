const lookupForm = document.getElementById("lookup-form");
const keyInput = document.getElementById("key-input");
const statusArea = document.getElementById("status-area");
const progressSteps = document.getElementById("progress-steps");
const progressArea = document.getElementById("progress-area");
const progressSummary = document.getElementById("progress-summary");
const toggleActivityBtn = document.getElementById("toggle-activity");
const activityLog = document.getElementById("activity-log");
const reviewSummaryCard = document.getElementById("review-summary-card");
const reviewItems = document.getElementById("review-items");
const citationCard = document.getElementById("citation-card");
const verificationSection = document.getElementById("verification-code-section");
const annotationModalRoot = document.getElementById("annotation-modal-root");
const reviewCard = document.getElementById("review-card");
const reviewContent = document.getElementById("review-content");
const pdfDownload = document.getElementById("pdf-download");

let pollingInterval = null;
let activityExpanded = false;
let progressFetchInFlight = false;
let currentKey = null;
let parsedReviewData = null;

toggleActivityBtn.addEventListener("click", () => {
  activityExpanded = !activityExpanded;
  activityLog.style.display = activityExpanded ? "block" : "none";
  toggleActivityBtn.textContent = activityExpanded ? "Hide Agent Activity" : "View Agent Activity";
});

// Auto-fill key from URL query param (?review_id= preferred, ?key= for backward compat)
const params = new URLSearchParams(window.location.search);
const urlKey = params.get("review_id") || params.get("key");
if (urlKey) {
  keyInput.value = urlKey;
  checkStatus(urlKey);
}

lookupForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const key = keyInput.value.trim();
  if (key) checkStatus(key);
});

// =========================================================
// Status checking
// =========================================================
async function checkStatus(key) {
  currentKey = key;
  // Update browser URL to shareable link
  const newUrl = new URL(window.location);
  newUrl.searchParams.set("review_id", key);
  newUrl.searchParams.delete("key");
  history.replaceState(null, "", newUrl);
  stopPolling();
  statusArea.style.display = "block";
  statusArea.innerHTML = `<div class="message info"><span class="spinner"></span> Checking status...</div>`;
  reviewCard.style.display = "none";
  reviewSummaryCard.style.display = "none";
  reviewItems.innerHTML = "";
  citationCard.style.display = "none";
  verificationSection.style.display = "none";
  annotationModalRoot.innerHTML = "";

  try {
    const resp = await fetch(`${API_BASE_URL}/api/status/${key}`);
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Not found.");
    }

    const data = await resp.json();
    renderStatus(data);
    updateProgressSteps(data.status);

    if (data.status === "ocr" || data.status === "reviewing") {
      fetchProgress(key);
    } else {
      progressArea.style.display = "none";
    }

    if (data.status === "completed") {
      await fetchReview(key);
    } else if (data.status !== "failed") {
      startPolling(key);
    }
  } catch (err) {
    statusArea.innerHTML = `<div class="message error"><strong>Error:</strong> ${err.message}</div>`;
    progressSteps.style.display = "none";
  }
}

function renderStatus(data) {
  const isByok = data.mode === "byok";
  const modeLabel = isByok ? "BYOK" : "Queue";
  const pendingLabel = isByok ? "Pending — priority processing" : "Pending — waiting in queue";

  const statusLabels = {
    pending: pendingLabel,
    ocr: "OCR — extracting text from PDF",
    reviewing: "Reviewing — AI agent is analyzing your paper",
    completed: "Completed — review is ready",
    failed: "Failed — an error occurred",
  };

  let html = `
    <div class="message info">
      <strong>File:</strong> ${escapeHtml(data.filename)}<br>
      <strong>Mode:</strong> ${modeLabel}<br>
      <strong>Status:</strong> <span class="status-badge status-${data.status}">${data.status}</span>
      &mdash; <em>${statusLabels[data.status] || data.status}</em>
    </div>`;

  if (data.status === "failed" && data.error_message) {
    html += `<div class="message error"><strong>Error details:</strong> ${escapeHtml(data.error_message)}</div>`;
  }

  if (data.status !== "completed" && data.status !== "failed") {
    html += `<div class="message info" style="margin-top:0.5rem;"><span class="spinner"></span> Auto-refreshing every 30 seconds...</div>`;
  }

  statusArea.innerHTML = html;
}

// =========================================================
// Progress Steps
// =========================================================
const STEP_ORDER = ["pending", "ocr", "reviewing", "completed"];

function updateProgressSteps(currentStatus) {
  if (currentStatus === "failed") {
    progressSteps.style.display = "none";
    return;
  }

  progressSteps.style.display = "flex";
  const currentIdx = STEP_ORDER.indexOf(currentStatus);

  document.querySelectorAll(".progress-step").forEach((el) => {
    const stepName = el.dataset.step;
    const stepIdx = STEP_ORDER.indexOf(stepName);

    el.classList.remove("done", "active");
    if (stepIdx < currentIdx) {
      el.classList.add("done");
    } else if (stepIdx === currentIdx) {
      if (currentStatus === "completed") {
        el.classList.add("done");
      } else {
        el.classList.add("active");
      }
    }
  });
}

// =========================================================
// Progress / Activity
// =========================================================
async function fetchProgress(key) {
  if (progressFetchInFlight) return;
  progressFetchInFlight = true;
  try {
    const resp = await fetch(`${API_BASE_URL}/api/status/${key}/progress`);
    if (!resp.ok) return;

    const data = await resp.json();
    if (data.total_steps === 0) {
      progressArea.style.display = "none";
      return;
    }

    progressArea.style.display = "block";

    let summaryHtml = `<div class="progress-info"><span class="spinner"></span> <strong>${data.total_steps}</strong> steps processed`;
    if (data.last_action_summary) {
      const truncated = data.last_action_summary.length > 150
        ? data.last_action_summary.substring(0, 150) + "..."
        : data.last_action_summary;
      summaryHtml += ` &mdash; <em>${escapeHtml(truncated)}</em>`;
    }
    summaryHtml += `</div>`;
    progressSummary.innerHTML = summaryHtml;

    if (data.events && data.events.length > 0) {
      toggleActivityBtn.style.display = "inline-block";

      let logHtml = "";
      for (const ev of data.events) {
        const time = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "";
        const tool = ev.tool_name ? `<span class="event-tool">${escapeHtml(ev.tool_name)}</span>` : "";
        const summary = ev.summary
          ? escapeHtml(ev.summary.length > 200 ? ev.summary.substring(0, 200) + "..." : ev.summary)
          : "";
        logHtml += `<div class="event-item">
          <span class="event-step">#${ev.step}</span>
          ${time ? `<span class="event-time">${time}</span>` : ""}
          ${tool}
          ${summary ? `<span class="event-summary">${summary}</span>` : ""}
        </div>`;
      }
      activityLog.innerHTML = logHtml;
    } else {
      toggleActivityBtn.style.display = "none";
    }
  } catch {
    // Silently ignore progress fetch errors
  } finally {
    progressFetchInFlight = false;
  }
}

// =========================================================
// Review Parsing & Rendering
// =========================================================

function parseReviewMarkdown(md) {
  const result = { items: [], citations: [], raw: md };

  try {
    // Extract citation list — greedy to end of string
    const citationMatch = md.match(/^#{1,4}\s*(?:Citation\s*List|References|Citations|Cited Papers)[^\n]*\n([\s\S]*)/mi);
    if (citationMatch) {
      const citBlock = citationMatch[1].trim();
      result.citations = citBlock
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.length > 0);
    }

    // Extract items: ## Item N: Title
    const itemPattern = /^##\s*Item\s+(\d+)\s*:\s*(.+)/gmi;
    const sections = [];
    let match;
    while ((match = itemPattern.exec(md)) !== null) {
      sections.push({ number: parseInt(match[1], 10), title: match[2].trim(), startIdx: match.index });
    }

    for (let i = 0; i < sections.length; i++) {
      const sec = sections[i];
      const endIdx = i + 1 < sections.length ? sections[i + 1].startIdx : md.length;
      const body = md.substring(sec.startIdx, endIdx);

      const item = {
        number: sec.number,
        title: sec.title,
        mainCriticism: "",
        evalCriteria: "",
        evidence: [],
      };

      // Parse Claim section
      const claimMatch = body.match(/####\s*Claim\s*\n([\s\S]*?)(?=####|$)/i);
      if (claimMatch) {
        const claimText = claimMatch[1];
        const critMatch = claimText.match(/\*\s*Main point of criticism\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (critMatch) item.mainCriticism = critMatch[1].trim();
        const evalMatch = claimText.match(/\*\s*Evaluation criteria\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (evalMatch) item.evalCriteria = evalMatch[1].trim();
      }

      // Parse Evidence section
      const evidenceMatch = body.match(/####\s*Evidence\s*\n([\s\S]*?)(?=####|##\s(?!#)|$)/i);
      if (evidenceMatch) {
        const evText = evidenceMatch[1];
        const quoteBlocks = evText.split(/(?=\*\s*Quote\s*:)/i);
        for (const block of quoteBlocks) {
          const trimmed = block.trim();
          if (!trimmed) continue;
          const qMatch = trimmed.match(/\*\s*Quote\s*:\s*([\s\S]*?)(?=\n\s+\*\s*Comment\s*:|$)/i);
          const cMatch = trimmed.match(/\*\s*Comment\s*:\s*([\s\S]*?)$/i);
          if (qMatch) {
            item.evidence.push({
              quote: qMatch[1].trim(),
              comment: cMatch ? cMatch[1].trim() : "",
            });
          }
        }
      }

      result.items.push(item);
    }
  } catch {
    // Fallback to raw markdown display
  }

  return result;
}

function renderStructuredReview(parsed, key) {
  parsedReviewData = parsed;

  // -- Summary Card --
  reviewSummaryCard.style.display = "block";
  let summaryHtml = `
    <div class="summary-stats">
      <div class="stat-item">
        <span class="stat-number">${parsed.items.length}</span>
        <span class="stat-label">Review Items</span>
      </div>
      <div class="stat-item">
        <span class="stat-number">${parsed.citations.length}</span>
        <span class="stat-label">Citations</span>
      </div>
    </div>
    <div class="summary-items">
      ${parsed.items.map((item) =>
        `<span class="summary-tag" onclick="scrollToItem(${item.number})">${escapeHtml(item.title)}</span>`
      ).join("")}
    </div>`;
  reviewSummaryCard.innerHTML = summaryHtml;

  // -- Item Cards --
  let itemsHtml = "";
  for (const item of parsed.items) {
    const chevronSvg = `<svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
    const criteriaTag = item.evalCriteria
      ? `<span class="criteria-tag">${escapeHtml(item.evalCriteria)}</span>` : "";

    let bodyHtml = "";

    if (item.mainCriticism) {
      bodyHtml += `
        <div class="claim-box">
          <div class="claim-label">Main Point of Criticism</div>
          <div class="claim-text">${renderInlineMarkdown(item.mainCriticism)}</div>
        </div>`;
    }

    if (item.evalCriteria) {
      bodyHtml += `
        <div class="criteria-box">
          <span class="criteria-label">Evaluation Criteria:</span>
          <span class="criteria-value">${escapeHtml(item.evalCriteria)}</span>
        </div>`;
    }

    if (item.evidence.length > 0) {
      bodyHtml += `<div class="evidence-section">
        <div class="evidence-header">Evidence</div>`;

      for (let j = 0; j < item.evidence.length; j++) {
        const ev = item.evidence[j];
        bodyHtml += `
          <div class="evidence-pair">
            <div class="quote-box">
              <div class="quote-label">\u{1F4D6} Quote ${j + 1}</div>
              <div class="quote-text">${renderInlineMarkdown(ev.quote)}</div>
            </div>
            ${ev.comment ? `
            <div class="comment-box">
              <div class="comment-label">\u{1F916} Comment</div>
              <div class="comment-text">${renderInlineMarkdown(ev.comment)}</div>
            </div>` : ""}
          </div>`;
      }
      bodyHtml += `</div>`;
    }

    bodyHtml += `
        <button class="annotate-pill" onclick="event.stopPropagation(); openItemAnnotation(${item.number})">Provide us your opinion on this criticism!</button>`;

    itemsHtml += `
      <div class="review-item" id="review-item-${item.number}">
        <div class="review-item-header" onclick="toggleItem(this)">
          <div class="review-item-title">
            <span class="item-number">${item.number}</span>
            <span>${escapeHtml(item.title)}</span>
            ${criteriaTag}
          </div>
          ${chevronSvg}
        </div>
        <div class="review-item-body">${bodyHtml}</div>
      </div>`;
  }
  reviewItems.innerHTML = itemsHtml;

  // -- Citation Card --
  if (parsed.citations.length > 0) {
    citationCard.style.display = "block";
    citationCard.innerHTML = `
      <h3>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
        References (${parsed.citations.length})
      </h3>
      <ol class="citation-list">
        ${parsed.citations.map((c, i) => `<li id="ref${i + 1}"><a id="citation-${i + 1}"></a>${renderCitation(c)}</li>`).join("")}
      </ol>`;
  }

  // -- Action buttons row --
  const actionsRow = document.createElement("div");
  actionsRow.className = "summary-actions";

  const dlBtn = document.createElement("a");
  dlBtn.className = "btn btn-secondary btn-sm";
  dlBtn.style.display = "inline-flex";
  dlBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download PDF`;
  dlBtn.onclick = (e) => { e.preventDefault(); downloadPDF(key); };

  const shareBtn = document.createElement("button");
  shareBtn.className = "btn btn-secondary btn-sm";
  shareBtn.style.display = "inline-flex";
  shareBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg> Share the review with your co-author`;
  shareBtn.onclick = () => {
    const url = new URL(window.location);
    url.searchParams.set("review_id", key);
    navigator.clipboard.writeText(url.toString()).then(() => {
      const orig = shareBtn.innerHTML;
      shareBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Link copied!`;
      shareBtn.style.color = "var(--success)";
      shareBtn.style.borderColor = "var(--success-border)";
      setTimeout(() => { shareBtn.innerHTML = orig; shareBtn.style.color = ""; shareBtn.style.borderColor = ""; }, 2000);
    });
  };

  actionsRow.appendChild(dlBtn);
  actionsRow.appendChild(shareBtn);
  reviewSummaryCard.appendChild(actionsRow);

  // -- Fetch verification code + show annotation modal --
  fetchVerificationCode(key);
  showAnnotationModal(key, parsed.items, 0);
}

function renderRawReview(md, key) {
  reviewCard.style.display = "block";
  reviewContent.innerHTML = marked.parse(md);
  pdfDownload.style.display = "inline-flex";
  pdfDownload.onclick = (e) => { e.preventDefault(); downloadPDF(key); };
}

async function downloadPDF(key) {
  try {
    const pdfResp = await fetch(`${API_BASE_URL}/api/review/${key}/pdf`);
    if (!pdfResp.ok) throw new Error("PDF not available.");
    const blob = await pdfResp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `review_${key}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Failed to download PDF: " + err.message);
  }
}

// =========================================================
// Fetch and display review
// =========================================================
async function fetchReview(key) {
  try {
    const resp = await fetch(`${API_BASE_URL}/api/review/${key}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to load review.");
    }

    const data = await resp.json();
    if (data.review_markdown) {
      const parsed = parseReviewMarkdown(data.review_markdown);

      if (parsed.items.length > 0) {
        renderStructuredReview(parsed, key);
      } else {
        renderRawReview(data.review_markdown, key);
      }
    } else {
      reviewCard.style.display = "block";
      reviewContent.innerHTML = `<div class="message error">Review completed but content is unavailable.</div>`;
    }
  } catch (err) {
    reviewCard.style.display = "block";
    reviewContent.innerHTML = `<div class="message error"><strong>Error loading review:</strong> ${err.message}</div>`;
  }
}

// =========================================================
// Verification Code
// =========================================================
let verificationCodeFiles = [];
let verificationExpanded = false;

async function fetchVerificationCode(key) {
  try {
    const resp = await fetch(`${API_BASE_URL}/api/review/${key}/verification-code`);
    if (!resp.ok) return;
    const data = await resp.json();
    verificationCodeFiles = data.files || [];

    if (verificationCodeFiles.length === 0) return;

    verificationSection.style.display = "block";
    verificationSection.innerHTML = `
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <h2 style="margin-bottom:0;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
            Reviewer's Verification Code
          </h2>
          <button class="btn btn-secondary btn-sm" id="toggle-vcode">Display Code</button>
        </div>
        <p style="margin-top:0.5rem;">The AI reviewer generated ${verificationCodeFiles.length} file(s) to verify claims in the paper.</p>
        <div id="vcode-viewer" style="display:none;"></div>
      </div>`;

    document.getElementById("toggle-vcode").addEventListener("click", async () => {
      verificationExpanded = !verificationExpanded;
      const viewer = document.getElementById("vcode-viewer");
      const btn = document.getElementById("toggle-vcode");

      if (verificationExpanded) {
        btn.textContent = "Hide Code";
        viewer.style.display = "block";
        await loadVerificationCodeFiles(key);
      } else {
        btn.textContent = "Display Code";
        viewer.style.display = "none";
      }
    });
  } catch {
    // Silently ignore
  }
}

async function loadVerificationCodeFiles(key) {
  const viewer = document.getElementById("vcode-viewer");
  let html = "";

  for (const file of verificationCodeFiles) {
    try {
      const resp = await fetch(`${API_BASE_URL}/api/review/${key}/verification-code/${file.name}`);
      if (!resp.ok) continue;
      const data = await resp.json();

      html += `
        <div class="code-viewer">
          <div class="code-file-header">
            <span>${escapeHtml(file.name)}</span>
            <span>${formatBytes(file.size)}</span>
          </div>
          <div class="code-file-content">
            <pre>${escapeHtml(data.content)}</pre>
          </div>
        </div>`;
    } catch {
      html += `
        <div class="code-viewer">
          <div class="code-file-header"><span>${escapeHtml(file.name)}</span></div>
          <div class="code-file-content"><pre>(Unable to load file content)</pre></div>
        </div>`;
    }
  }

  viewer.innerHTML = html;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// =========================================================
// Annotation Modal
// =========================================================

// Generate a persistent annotator ID per browser
function getAnnotatorId() {
  let id = localStorage.getItem("annotator_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("annotator_id", id);
  }
  return id;
}

let annotationState = { correctness: null, significance: null, evidence_quality: null };
let existingAnnotations = [];

function showAnnotationModal(key, items, currentIndex) {
  if (!items || currentIndex >= items.length) {
    closeAnnotationModal();
    return;
  }

  const item = items[currentIndex];

  // Reset state for new item
  annotationState = { correctness: null, significance: null, evidence_quality: null };

  // Load existing annotations once, then render
  const render = () => {
    const existingForItem = existingAnnotations.find(a => a.item_number === item.number);
    if (existingForItem) {
      annotationState = {
        correctness: existingForItem.correctness,
        significance: existingForItem.significance,
        evidence_quality: existingForItem.evidence_quality,
      };
    }

    annotationModalRoot.innerHTML = `
      <div class="modal-overlay" id="annotation-overlay">
        <div class="modal">
          <h3>Rate Review Item ${currentIndex + 1} of ${items.length}</h3>
          <p class="modal-subtitle">Help us evaluate the quality of this AI-generated review.</p>

          <div class="item-preview">
            <strong>${escapeHtml(item.title)}</strong>
            ${item.mainCriticism ? `<div style="margin-top:0.5rem;"><span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--cmu-red);">Main Point of Criticism</span><br>${renderInlineMarkdown(item.mainCriticism)}</div>` : ""}
            ${item.evidence.length > 0 ? `<div style="margin-top:0.75rem;border-top:1px solid rgba(196,18,48,0.15);padding-top:0.6rem;"><span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--gray-500);">Evidence</span>${item.evidence.map((ev, j) => `<div style="margin-top:0.5rem;"><div style="background:#fff;border:1px solid var(--gray-200);border-radius:6px;padding:0.5rem 0.7rem;font-style:italic;font-size:0.85rem;color:var(--gray-600);">${renderInlineMarkdown(ev.quote)}</div>${ev.comment ? `<div style="border-left:2px solid var(--gray-300);margin-left:0.75rem;margin-top:0.3rem;padding:0.3rem 0.6rem;font-size:0.85rem;color:var(--gray-700);">${renderInlineMarkdown(ev.comment)}</div>` : ""}</div>`).join("")}</div>` : ""}
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Is this criticism correct?</div>
            <div class="annotation-buttons" data-field="correctness">
              <button class="annotation-btn ${annotationState.correctness === 'correct' ? 'selected' : ''}" data-value="correct">Correct</button>
              <button class="annotation-btn ${annotationState.correctness === 'incorrect' ? 'selected' : ''}" data-value="incorrect">Incorrect</button>
            </div>
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Does it touch a significant issue?</div>
            <div class="annotation-buttons" data-field="significance">
              <button class="annotation-btn ${annotationState.significance === 'significant' ? 'selected' : ''}" data-value="significant">Significant</button>
              <button class="annotation-btn ${annotationState.significance === 'marginally_significant' ? 'selected' : ''}" data-value="marginally_significant">Marginally Significant</button>
              <button class="annotation-btn ${annotationState.significance === 'not_significant' ? 'selected' : ''}" data-value="not_significant">Not Significant</button>
            </div>
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Is the evidence sufficient?</div>
            <div class="annotation-buttons" data-field="evidence_quality">
              <button class="annotation-btn ${annotationState.evidence_quality === 'sufficient' ? 'selected' : ''}" data-value="sufficient">Sufficient</button>
              <button class="annotation-btn ${annotationState.evidence_quality === 'insufficient' ? 'selected' : ''}" data-value="insufficient">Insufficient</button>
            </div>
          </div>

          <div class="modal-actions">
            <button class="btn btn-secondary btn-sm" id="annotation-skip">Skip</button>
            <button class="btn btn-primary btn-sm" id="annotation-submit" style="margin-top:0;">Submit</button>
          </div>
        </div>
      </div>`;

    // Wire up buttons
    document.querySelectorAll(".annotation-buttons").forEach(group => {
      const field = group.dataset.field;
      group.querySelectorAll(".annotation-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          group.querySelectorAll(".annotation-btn").forEach(b => b.classList.remove("selected"));
          btn.classList.add("selected");
          annotationState[field] = btn.dataset.value;
        });
      });
    });

    // Skip closes modal (no auto-advance)
    document.getElementById("annotation-skip").addEventListener("click", closeAnnotationModal);

    // Submit saves and advances to next item
    document.getElementById("annotation-submit").addEventListener("click", async () => {
      try {
        await fetch(`${API_BASE_URL}/api/review/${key}/annotations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            item_number: item.number,
            annotator_id: getAnnotatorId(),
            ...annotationState,
          }),
        });
      } catch {
        // Ignore errors silently
      }
      // Advance to next item
      showAnnotationModal(key, items, currentIndex + 1);
    });

    // Close on overlay click (no auto-advance)
    document.getElementById("annotation-overlay").addEventListener("click", (e) => {
      if (e.target === e.currentTarget) closeAnnotationModal();
    });
  };

  // Fetch existing annotations on first call, reuse cache after
  if (existingAnnotations.length === 0 && currentIndex === 0) {
    fetch(`${API_BASE_URL}/api/review/${key}/annotations?annotator_id=${encodeURIComponent(getAnnotatorId())}`)
      .then(r => r.json())
      .then(data => { existingAnnotations = data; render(); })
      .catch(() => render());
  } else {
    render();
  }
}

function showSingleAnnotationModal(key, item) {
  // Open modal for a single item (no auto-advance)
  annotationState = { correctness: null, significance: null, evidence_quality: null };

  fetch(`${API_BASE_URL}/api/review/${key}/annotations?annotator_id=${encodeURIComponent(getAnnotatorId())}`)
    .then(r => r.json())
    .then(data => {
      existingAnnotations = data;
      const existingForItem = data.find(a => a.item_number === item.number);
      if (existingForItem) {
        annotationState = {
          correctness: existingForItem.correctness,
          significance: existingForItem.significance,
          evidence_quality: existingForItem.evidence_quality,
        };
      }
      renderSingleAnnotationModal(key, item);
    })
    .catch(() => renderSingleAnnotationModal(key, item));
}

function renderSingleAnnotationModal(key, item) {
  annotationModalRoot.innerHTML = `
    <div class="modal-overlay" id="annotation-overlay">
      <div class="modal">
        <h3>Rate Review Item ${item.number}</h3>
        <p class="modal-subtitle">Rate this specific review item.</p>

        <div class="item-preview">
          <strong>${escapeHtml(item.title)}</strong>
          ${item.mainCriticism ? `<div style="margin-top:0.5rem;"><span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--cmu-red);">Main Point of Criticism</span><br>${renderInlineMarkdown(item.mainCriticism)}</div>` : ""}
          ${item.evidence.length > 0 ? `<div style="margin-top:0.75rem;border-top:1px solid rgba(196,18,48,0.15);padding-top:0.6rem;"><span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--gray-500);">Evidence</span>${item.evidence.map((ev, j) => `<div style="margin-top:0.5rem;"><div style="background:#fff;border:1px solid var(--gray-200);border-radius:6px;padding:0.5rem 0.7rem;font-style:italic;font-size:0.85rem;color:var(--gray-600);">${renderInlineMarkdown(ev.quote)}</div>${ev.comment ? `<div style="border-left:2px solid var(--gray-300);margin-left:0.75rem;margin-top:0.3rem;padding:0.3rem 0.6rem;font-size:0.85rem;color:var(--gray-700);">${renderInlineMarkdown(ev.comment)}</div>` : ""}</div>`).join("")}</div>` : ""}
        </div>

        <div class="annotation-group">
          <div class="annotation-group-label">Is this criticism correct?</div>
          <div class="annotation-buttons" data-field="correctness">
            <button class="annotation-btn ${annotationState.correctness === 'correct' ? 'selected' : ''}" data-value="correct">Correct</button>
            <button class="annotation-btn ${annotationState.correctness === 'incorrect' ? 'selected' : ''}" data-value="incorrect">Incorrect</button>
          </div>
        </div>

        <div class="annotation-group">
          <div class="annotation-group-label">Does it touch a significant issue?</div>
          <div class="annotation-buttons" data-field="significance">
            <button class="annotation-btn ${annotationState.significance === 'significant' ? 'selected' : ''}" data-value="significant">Significant</button>
            <button class="annotation-btn ${annotationState.significance === 'marginally_significant' ? 'selected' : ''}" data-value="marginally_significant">Marginally Significant</button>
            <button class="annotation-btn ${annotationState.significance === 'not_significant' ? 'selected' : ''}" data-value="not_significant">Not Significant</button>
          </div>
        </div>

        <div class="annotation-group">
          <div class="annotation-group-label">Is the evidence sufficient?</div>
          <div class="annotation-buttons" data-field="evidence_quality">
            <button class="annotation-btn ${annotationState.evidence_quality === 'sufficient' ? 'selected' : ''}" data-value="sufficient">Sufficient</button>
            <button class="annotation-btn ${annotationState.evidence_quality === 'insufficient' ? 'selected' : ''}" data-value="insufficient">Insufficient</button>
          </div>
        </div>

        <div class="modal-actions">
          <button class="btn btn-secondary btn-sm" id="annotation-skip">Cancel</button>
          <button class="btn btn-primary btn-sm" id="annotation-submit" style="margin-top:0;">Submit</button>
        </div>
      </div>
    </div>`;

  // Wire up buttons
  document.querySelectorAll(".annotation-buttons").forEach(group => {
    const field = group.dataset.field;
    group.querySelectorAll(".annotation-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        group.querySelectorAll(".annotation-btn").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        annotationState[field] = btn.dataset.value;
      });
    });
  });

  document.getElementById("annotation-skip").addEventListener("click", closeAnnotationModal);

  document.getElementById("annotation-submit").addEventListener("click", async () => {
    try {
      await fetch(`${API_BASE_URL}/api/review/${key}/annotations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          item_number: item.number,
          annotator_id: getAnnotatorId(),
          ...annotationState,
        }),
      });
    } catch {
      // Ignore errors silently
    }
    closeAnnotationModal();
  });

  document.getElementById("annotation-overlay").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeAnnotationModal();
  });
}

function closeAnnotationModal() {
  annotationModalRoot.innerHTML = "";
}

// =========================================================
// Helpers
// =========================================================
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderInlineMarkdown(text) {
  // Handle nested-bracket citation links first: [[1]](#ref1) or [[1]](#citation-1)
  let result = text.replace(/\[\[(\d+)\]\]\((#[^)]+)\)/g, (match, num, url) => {
    return `<a href="${url}" class="citation-ref" style="scroll-behavior:smooth;">[${num}]</a>`;
  });
  // Then handle standard markdown links: [text](url)
  result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
    if (url.startsWith("#")) {
      return `<a href="${url}" style="scroll-behavior:smooth;">${label}</a>`;
    }
    return `<a href="${url}" target="_blank" rel="noopener">${label}</a>`;
  });
  // Auto-link plain [N] citation references that weren't already converted
  result = result.replace(/(?<![<\w])(?<!\[)\[(\d+)\](?!\()/g, (match, num) => {
    return `<a href="#ref${num}" class="citation-ref" style="scroll-behavior:smooth;">[${num}]</a>`;
  });
  result = result
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
  return result;
}

function renderCitation(text) {
  const stripped = text.replace(/^\[\d+\]\s*/, "");
  return stripped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function toggleItem(header) {
  const item = header.closest(".review-item");
  item.classList.toggle("open");
}

function openItemAnnotation(itemNumber) {
  if (!parsedReviewData || !currentKey) return;
  const item = parsedReviewData.items.find(i => i.number === itemNumber);
  if (item) showSingleAnnotationModal(currentKey, item);
}

function scrollToItem(number) {
  const el = document.getElementById(`review-item-${number}`);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("open");
    el.style.boxShadow = "0 0 0 2px var(--cmu-red)";
    setTimeout(() => { el.style.boxShadow = ""; }, 1500);
  }
}

function startPolling(key) {
  pollingInterval = setInterval(() => checkStatus(key), 30000);
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
}
