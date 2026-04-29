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
    if (data.error_message.includes("[OUT_OF_BUDGET]")) {
      html += `
        <div class="message error" style="text-align:center;padding:1.5rem;">
          <div style="font-size:1.5rem;margin-bottom:0.5rem;">&#9888;</div>
          <strong>We're currently out of API credit and will adjust this ASAP.</strong>
          <p style="margin-top:0.75rem;font-size:0.88rem;">Your submission has been saved. It will be processed automatically once credit is restored.</p>
          <p style="margin-top:0.75rem;font-size:0.88rem;">If you're interested in funding this project, please contact:<br>
            <a href="mailto:seungone@cmu.edu" style="font-weight:600;">seungone@cmu.edu</a> and
            <a href="mailto:gneubig@cs.cmu.edu" style="font-weight:600;">gneubig@cs.cmu.edu</a>
          </p>
        </div>`;
    } else {
      html += `<div class="message error"><strong>Error details:</strong> ${escapeHtml(data.error_message)}</div>`;
    }
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

      const toolLabels = {
        file_editor: "file",
        terminal: "terminal",
        tavily__search: "search",
        tavily__extract: "extract",
        think: "think",
        task_tracker: "task",
        finish: "finish",
      };
      let logHtml = "";
      for (const ev of data.events) {
        const time = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : "";
        const label = (ev.tool_name && toolLabels[ev.tool_name]) || ev.tool_name || "";
        const tool = label ? `<span class="event-tool">${escapeHtml(label)}</span>` : "";
        const summary = ev.summary
          ? escapeHtml(ev.summary.length > 250 ? ev.summary.substring(0, 250) + "..." : ev.summary)
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
        limitationsStatus: "",
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
        const limMatch = claimText.match(/\*\s*Limitations status\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (limMatch) item.limitationsStatus = limMatch[1].trim();
      }

      // Parse Evidence section
      const evidenceMatch = body.match(/####\s*Evidence\s*\n([\s\S]*?)(?=####|##\s(?!#)|$)/i);
      if (evidenceMatch) {
        let evText = evidenceMatch[1];
        // Normalize bold labels: "* **Quote**:" → "* Quote:"
        evText = evText.replace(/\*\s*\*\*Quote\*\*/g, "* Quote");
        evText = evText.replace(/\*\s*\*\*Comment\*\*/g, "* Comment");
        const quoteBlocks = evText.split(/(?=\*\s*Quote(?:\s+from\s+[^:]+)?\s*:)/i);
        for (const block of quoteBlocks) {
          const trimmed = block.trim();
          if (!trimmed) continue;
          // Match quote text up to comment (indented or same level)
          const qMatch = trimmed.match(/\*\s*Quote(?:\s+from\s+[^:]+)?\s*:\s*([\s\S]*?)(?=\n\s*\*\s*Comment\s*:|$)/i);
          const cMatch = trimmed.match(/\*\s*Comment\s*:\s*([\s\S]*?)$/i);
          if (qMatch) {
            item.evidence.push({
              quote: qMatch[1].trim(),
              comment: cMatch ? cMatch[1].trim() : "",
            });
          }
        }
      }

      // Parse Concrete Action Item section
      const actionMatch = body.match(/####\s*Concrete Action Item\s*\n([\s\S]*?)(?=####|##\s(?!#)|$)/i);
      if (actionMatch) {
        const aiText = actionMatch[1];
        const ai = {};
        const typeM = aiText.match(/\*\s*Action type\s*:\s*(.+)/i);
        if (typeM) ai.actionType = typeM[1].trim();
        const origM = aiText.match(/\*\s*Original text\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (origM) ai.originalText = origM[1].trim();
        const suggM = aiText.match(/\*\s*Suggested text\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (suggM) ai.suggestedText = suggM[1].trim();
        const locM = aiText.match(/\*\s*Location\s*:\s*(.+)/i);
        if (locM) ai.location = locM[1].trim();
        const paraM = aiText.match(/\*\s*New paragraph\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (paraM) ai.newParagraph = paraM[1].trim();
        const descM = aiText.match(/\*\s*Description\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (descM) ai.description = descM[1].trim();
        const codeM = aiText.match(/\*\s*Key code changes\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (codeM) ai.keyCodeChanges = codeM[1].trim();
        const filesM = aiText.match(/\*\s*Files modified\s*:\s*(.+)/i);
        if (filesM) ai.filesModified = filesM[1].trim();
        const runM = aiText.match(/\*\s*Run command\s*:\s*([\s\S]*?)(?=\n\*\s|\n####|$)/i);
        if (runM) ai.runCommand = runM[1].trim();
        item.actionItem = ai;
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

    if (item.limitationsStatus) {
      const isNotMentioned = item.limitationsStatus.toLowerCase().includes("not mentioned");
      const badgeClass = isNotMentioned ? "limitations-badge not-mentioned" : "limitations-badge mentioned-not-justifiable";
      bodyHtml += `
        <div class="limitations-box">
          <span class="limitations-label">Limitations Status:</span>
          <span class="${badgeClass}">${escapeHtml(item.limitationsStatus)}</span>
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

    if (item.actionItem) {
      const ai = item.actionItem;
      const isWriting = ai.actionType && ai.actionType.toLowerCase().includes("fix the writing");
      const badgeClass = isWriting ? "action-badge-writing" : "action-badge-implementation";
      const badgeText = isWriting ? "Fix the writing" : "Add new implementation";

      bodyHtml += `
        <div class="action-item-section">
          <div class="action-item-header" onclick="toggleActionItem(this)">
            <span class="action-item-label">Concrete Action Item</span>
            <span class="action-type-badge ${badgeClass}">${escapeHtml(badgeText)}</span>
            <svg class="chevron action-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
          <div class="action-item-body">`;

      if (isWriting) {
        if (ai.originalText && ai.suggestedText) {
          bodyHtml += `
            <div class="diff-view">
              <div class="diff-panel diff-original">
                <div class="diff-panel-header">Original</div>
                <div class="diff-panel-content">${renderInlineMarkdown(ai.originalText)}</div>
              </div>
              <div class="diff-panel diff-suggested">
                <div class="diff-panel-header">Suggested</div>
                <div class="diff-panel-content">${renderInlineMarkdown(ai.suggestedText)}</div>
              </div>
            </div>`;
        }
        if (ai.newParagraph) {
          bodyHtml += `
            <div class="new-paragraph-box">
              <div class="new-paragraph-label">New paragraph to add</div>
              <div class="new-paragraph-text">${renderInlineMarkdown(ai.newParagraph)}</div>
            </div>`;
        }
        if (ai.location) {
          bodyHtml += `<div class="action-location"><strong>Location:</strong> ${escapeHtml(ai.location)}</div>`;
        }
      } else {
        if (ai.description) {
          bodyHtml += `<div class="action-description">${renderInlineMarkdown(ai.description)}</div>`;
        }
        if (ai.keyCodeChanges) {
          bodyHtml += `
            <div class="code-changes-box">
              <div class="code-changes-header">Key Code Changes <button class="copy-btn" onclick="event.stopPropagation(); copyToClipboard(this, ${JSON.stringify(JSON.stringify(ai.keyCodeChanges))})">Copy</button></div>
              <pre class="code-block"><code>${escapeHtml(ai.keyCodeChanges)}</code></pre>
            </div>`;
        }
        if (ai.filesModified) {
          bodyHtml += `<div class="action-files"><strong>Files modified:</strong> ${escapeHtml(ai.filesModified)}</div>`;
        }
        if (ai.runCommand) {
          bodyHtml += `
            <div class="run-command-box">
              <div class="run-command-header">Run Command <button class="copy-btn" onclick="event.stopPropagation(); copyToClipboard(this, ${JSON.stringify(JSON.stringify(ai.runCommand))})">Copy</button></div>
              <pre class="code-block"><code>${escapeHtml(ai.runCommand)}</code></pre>
            </div>`;
        }
      }

      bodyHtml += `</div></div>`;
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

  // -- Verification Code (displayed before references) --
  fetchVerificationCodeInline(key);

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
  dlBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download Review (ZIP)`;
  dlBtn.onclick = (e) => { e.preventDefault(); downloadBundle(key); };

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

  // -- Typeset math (KaTeX) after content is in the DOM --
  // Use a small delay to ensure KaTeX auto-render script has loaded
  setTimeout(() => {
    typsetMath(reviewSummaryCard);
    typsetMath(reviewItems);
    typsetMath(citationCard);
  }, 100);

  // -- Show annotation modal (only once per page load) --
  if (!window._annotationModalShown) {
    window._annotationModalShown = true;
    showAnnotationModal(key, parsed.items, 0);
  }
}

function renderRawReview(md, key) {
  reviewCard.style.display = "block";
  reviewContent.innerHTML = marked.parse(md);
  pdfDownload.style.display = "inline-flex";
  pdfDownload.onclick = (e) => { e.preventDefault(); downloadPDF(key); };
}

async function downloadBundle(key) {
  try {
    const resp = await fetch(`${API_BASE_URL}/api/review/${key}/download`);
    if (!resp.ok) throw new Error("Download not available.");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `review_${key}.zip`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Failed to download: " + err.message);
  }
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

async function fetchVerificationCodeInline(key) {
  try {
    const resp = await fetch(`${API_BASE_URL}/api/review/${key}/verification-code`);
    if (!resp.ok) return;
    const data = await resp.json();
    verificationCodeFiles = data.files || [];

    if (verificationCodeFiles.length === 0) return;

    // Show verification code section before references (it's placed before citation-card in DOM)
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

async function fetchVerificationCode(key) {
  // Legacy - now handled by fetchVerificationCodeInline
  return fetchVerificationCodeInline(key);
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

let annotationState = { correctness: null, significance: null, evidence_quality: null, action_item_quality: null, free_text: "" };
let existingAnnotations = [];

function showAnnotationModal(key, items, currentIndex) {
  if (!items || currentIndex >= items.length) {
    closeAnnotationModal();
    return;
  }

  const item = items[currentIndex];

  // Reset state for new item
  annotationState = { correctness: null, significance: null, evidence_quality: null, action_item_quality: null, free_text: "" };

  // Load existing annotations once, then render
  const render = () => {
    const existingForItem = existingAnnotations.find(a => a.item_number === item.number);
    if (existingForItem) {
      annotationState = {
        correctness: existingForItem.correctness,
        significance: existingForItem.significance,
        evidence_quality: existingForItem.evidence_quality,
        action_item_quality: existingForItem.action_item_quality,
        free_text: existingForItem.free_text || "",
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
            <div class="annotation-hint">For each item, first click whether the main point of the item (criticism) is correct and clearly stated. This is a binary yes/no question: if you think there is any slight doubt, you can select "Incorrect", whereas "Correct" means every aspect of the main point is correct and clearly stated.</div>
            <div class="annotation-buttons" data-field="correctness">
              <button class="annotation-btn ${annotationState.correctness === 'correct' ? 'selected' : ''}" data-value="correct">Correct</button>
              <button class="annotation-btn ${annotationState.correctness === 'incorrect' ? 'selected' : ''}" data-value="incorrect">Incorrect</button>
            </div>
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Does it touch a significant issue?</div>
            <div class="annotation-hint">If you have clicked "Correct" above, you should then click whether the main point of the criticism is talking about a significant aspect of the paper that is constructive to enhance the paper rather than touching a minor issue. You could click on one of the three options:<br>
            <strong>Significant</strong> \u2014 items that you would consider to be insightful and helpful to improve the paper.<br>
            <strong>Marginally Significant</strong> \u2014 items that aren\u2019t helpful at all for improving the paper but are still worth remaining in the review itself. This includes issues like typos, stylistic issues, and suggestions for submitting to a different journal, etc.<br>
            <strong>Not Significant (Very marginal issue)</strong> \u2014 very minor items that should not affect the acceptance of this paper and are better to be removed from the review.</div>
            <div class="annotation-buttons" data-field="significance">
              <button class="annotation-btn ${annotationState.significance === 'significant' ? 'selected' : ''}" data-value="significant">Significant</button>
              <button class="annotation-btn ${annotationState.significance === 'marginally_significant' ? 'selected' : ''}" data-value="marginally_significant">Marginally Significant</button>
              <button class="annotation-btn ${annotationState.significance === 'not_significant' ? 'selected' : ''}" data-value="not_significant">Not Significant</button>
            </div>
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Is the evidence sufficient?</div>
            <div class="annotation-hint">Additionally, we ask you to check if the evidence is sufficient to support the main point of the criticism. This means whether the evidence provided by the AI reviewer well justifies its main point of criticism. You could click on one of the two options: "Evidence is sufficient" or "Requires more evidence".</div>
            <div class="annotation-buttons" data-field="evidence_quality">
              <button class="annotation-btn ${annotationState.evidence_quality === 'sufficient' ? 'selected' : ''}" data-value="sufficient">Evidence is sufficient</button>
              <button class="annotation-btn ${annotationState.evidence_quality === 'insufficient' ? 'selected' : ''}" data-value="insufficient">Requires more evidence</button>
            </div>
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Is the action item helpful in improving this paper?</div>
            <div class="annotation-hint">Evaluate whether the concrete action item suggested by the AI reviewer is practical and would genuinely help improve the paper if followed.</div>
            <div class="annotation-buttons" data-field="action_item_quality">
              <button class="annotation-btn ${annotationState.action_item_quality === 'helpful_executable' ? 'selected' : ''}" data-value="helpful_executable">Helpful and executable</button>
              <button class="annotation-btn ${annotationState.action_item_quality === 'helpful_needs_modification' ? 'selected' : ''}" data-value="helpful_needs_modification">Helpful but is not executable</button>
              <button class="annotation-btn ${annotationState.action_item_quality === 'not_helpful' ? 'selected' : ''}" data-value="not_helpful">Not helpful</button>
            </div>
          </div>

          <div class="annotation-group">
            <div class="annotation-group-label">Additional comments</div>
            <div class="annotation-hint">If you have any specific comments regarding this review item, you can provide your thoughts here. Also, by writing this, you can debate with the AI reviewer on whether this review item is valid or not.</div>
            <textarea class="annotation-textarea" id="annotation-free-text" placeholder="Share any additional thoughts about this review item..." rows="3">${escapeHtml(annotationState.free_text || "")}</textarea>
          </div>

          <div class="debate-trigger">
            <button class="btn btn-secondary btn-sm debate-btn" id="debate-btn" disabled>I want to debate with AI about this (Beta \u{1F9EA})</button>
            <div class="annotation-hint" style="margin-top:0.4rem;">You can debate with AI about this item if you provide your opinions about this review item. Please fill in all the buttons and text form above.</div>
          </div>

          <div class="modal-actions">
            <button class="btn btn-secondary btn-sm" id="annotation-skip">Skip</button>
            <button class="btn btn-primary btn-sm" id="annotation-submit" style="margin-top:0;">Submit</button>
          </div>
        </div>
      </div>`;

    // Check if all annotation fields are filled (for enabling debate button)
    function updateDebateButton() {
      const debateBtn = document.getElementById("debate-btn");
      if (!debateBtn) return;
      const allFilled = annotationState.correctness
        && annotationState.significance
        && annotationState.evidence_quality
        && annotationState.action_item_quality
        && (annotationState.free_text || "").trim();
      debateBtn.disabled = !allFilled;
    }

    // Wire up buttons
    document.querySelectorAll(".annotation-buttons").forEach(group => {
      const field = group.dataset.field;
      group.querySelectorAll(".annotation-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          group.querySelectorAll(".annotation-btn").forEach(b => b.classList.remove("selected"));
          btn.classList.add("selected");
          annotationState[field] = btn.dataset.value;
          updateDebateButton();
        });
      });
    });

    // Wire free text
    const freeTextEl = document.getElementById("annotation-free-text");
    if (freeTextEl) freeTextEl.addEventListener("input", () => {
      annotationState.free_text = freeTextEl.value;
      updateDebateButton();
    });

    // Wire debate button
    document.getElementById("debate-btn")?.addEventListener("click", () => {
      const allFilled = annotationState.correctness
        && annotationState.significance
        && annotationState.evidence_quality
        && annotationState.action_item_quality
        && (annotationState.free_text || "").trim();
      if (!allFilled) {
        alert("Please fill in all annotation fields (correctness, significance, evidence, action item helpfulness, and additional comments) before starting a debate.");
        return;
      }
      openDebateChat(key, item);
    });

    // Initialize debate button state
    updateDebateButton();

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
  annotationState = { correctness: null, significance: null, evidence_quality: null, action_item_quality: null, free_text: "" };

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
          free_text: existingForItem.free_text || "",
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

        <div class="annotation-group">
          <div class="annotation-group-label">Is the action item helpful in improving this paper?</div>
          <div class="annotation-buttons" data-field="action_item_quality">
            <button class="annotation-btn ${annotationState.action_item_quality === 'helpful_executable' ? 'selected' : ''}" data-value="helpful_executable">Helpful and executable</button>
            <button class="annotation-btn ${annotationState.action_item_quality === 'helpful_needs_modification' ? 'selected' : ''}" data-value="helpful_needs_modification">Helpful but is not executable</button>
            <button class="annotation-btn ${annotationState.action_item_quality === 'not_helpful' ? 'selected' : ''}" data-value="not_helpful">Not helpful</button>
          </div>
        </div>

        <div class="annotation-group">
          <div class="annotation-group-label">Additional comments (optional)</div>
          <textarea class="annotation-textarea" id="annotation-free-text" placeholder="Share any additional thoughts about this review item..." rows="3">${escapeHtml(annotationState.free_text || "")}</textarea>
        </div>

        <div class="debate-trigger">
          <button class="btn btn-secondary btn-sm debate-btn" id="debate-btn">I want to debate with AI about this (Beta \u{1F9EA})</button>
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

  // Wire free text
  const freeTextEl = document.getElementById("annotation-free-text");
  if (freeTextEl) freeTextEl.addEventListener("input", () => { annotationState.free_text = freeTextEl.value; });

  // Wire debate button
  document.getElementById("debate-btn")?.addEventListener("click", () => { openDebateChat(key, item); });

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
// Action Item Helpers
// =========================================================
function toggleActionItem(header) {
  const section = header.closest(".action-item-section");
  section.classList.toggle("expanded");
}

function copyToClipboard(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = orig; }, 1500);
  });
}

// =========================================================
// Debate with AI
// =========================================================
let debateState = {
  sessionId: null,
  messages: [],
  turnCount: 0,
  status: "active",
  isStreaming: false,
  currentKey: null,
  currentItem: null,
};

async function openDebateChat(key, item) {
  debateState.currentKey = key;
  debateState.currentItem = item;

  // Start debate session via API
  try {
    const resp = await fetch(`${API_BASE_URL}/api/review/${key}/debate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_number: item.number,
        annotator_id: getAnnotatorId(),
        user_annotations: {
          correctness: annotationState.correctness || null,
          significance: annotationState.significance || null,
          evidence_quality: annotationState.evidence_quality || null,
          action_item_quality: annotationState.action_item_quality || null,
          free_text: (annotationState.free_text || "").trim() || null,
        },
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || "Failed to start debate.");
      return;
    }
    const data = await resp.json();
    debateState.sessionId = data.session_id;
    debateState.messages = [];
    debateState.turnCount = 0;
    debateState.status = "active";
    debateState.isStreaming = false;
  } catch {
    alert("Failed to start debate session.");
    return;
  }

  // Transform modal to debate layout
  const modal = document.querySelector(".modal");
  if (!modal) return;
  modal.classList.add("modal-debate");

  // Build left panel with item info
  let leftHtml = `<h3>Review Item ${item.number}</h3>`;
  leftHtml += `<div class="item-preview"><strong>${escapeHtml(item.title)}</strong>`;
  if (item.mainCriticism) {
    leftHtml += `<div style="margin-top:0.5rem;"><span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--cmu-red);">Main Point of Criticism</span><br>${renderInlineMarkdown(item.mainCriticism)}</div>`;
  }
  if (item.evidence.length > 0) {
    leftHtml += `<div style="margin-top:0.75rem;border-top:1px solid rgba(196,18,48,0.15);padding-top:0.6rem;"><span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--gray-500);">Evidence</span>`;
    item.evidence.forEach((ev, j) => {
      leftHtml += `<div style="margin-top:0.5rem;"><div style="background:#fff;border:1px solid var(--gray-200);border-radius:6px;padding:0.5rem 0.7rem;font-style:italic;font-size:0.85rem;color:var(--gray-600);">${renderInlineMarkdown(ev.quote)}</div>`;
      if (ev.comment) leftHtml += `<div style="border-left:2px solid var(--gray-300);margin-left:0.75rem;margin-top:0.3rem;padding:0.3rem 0.6rem;font-size:0.85rem;color:var(--gray-700);">${renderInlineMarkdown(ev.comment)}</div>`;
      leftHtml += `</div>`;
    });
    leftHtml += `</div>`;
  }
  leftHtml += `</div>`;

  modal.innerHTML = `
    <div class="debate-layout">
      <div class="debate-left">${leftHtml}</div>
      <div class="debate-right">
        <div class="debate-header">
          <span class="debate-title">Debate with AI (Beta \u{1F9EA})</span>
          <span class="turn-counter" id="turn-counter">Turn 0/20</span>
        </div>
        <div class="chat-messages" id="chat-messages">
          <div class="chat-bubble chat-assistant"><div class="chat-message-text">I believe this criticism is valid and important. Try to convince me otherwise! Make your argument.</div></div>
        </div>
        <div class="chat-input-area" id="chat-input-area">
          <textarea class="chat-input" id="chat-input" placeholder="Type your argument..." rows="2"></textarea>
          <button class="btn btn-primary btn-sm" id="chat-send">Send</button>
        </div>
        <div class="debate-conclusion" id="debate-conclusion" style="display:none;"></div>
      </div>
    </div>`;

  document.getElementById("chat-send").addEventListener("click", () => sendDebateMessage());
  document.getElementById("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendDebateMessage();
    }
  });
}

async function sendDebateMessage() {
  if (debateState.isStreaming || debateState.status !== "active") return;

  const input = document.getElementById("chat-input");
  const content = input.value.trim();
  if (!content) return;

  input.value = "";
  debateState.isStreaming = true;

  // Increment for user message
  debateState.turnCount++;
  document.getElementById("turn-counter").textContent = `Turn ${debateState.turnCount}/20`;

  // Add user message bubble
  appendChatMessage("user", content);

  // Add empty assistant bubble for streaming
  const assistantBubble = appendChatMessage("assistant", "");
  const textEl = assistantBubble.querySelector(".chat-message-text");

  // Disable input during streaming
  const sendBtn = document.getElementById("chat-send");
  const inputEl = document.getElementById("chat-input");
  if (sendBtn) sendBtn.disabled = true;
  if (inputEl) inputEl.disabled = true;

  try {
    const resp = await fetch(
      `${API_BASE_URL}/api/review/${debateState.currentKey}/debate/${debateState.sessionId}/message`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      }
    );

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullText = "";
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") continue;
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              fullText += parsed.content;
              textEl.innerHTML = renderInlineMarkdown(fullText);
              const chatEl = document.getElementById("chat-messages");
              chatEl.scrollTop = chatEl.scrollHeight;
            }
            if (parsed.error) {
              textEl.textContent = "Error: " + parsed.error;
            }
          } catch {}
        }
      }
    }

    // Increment for AI response
    debateState.turnCount++;
    document.getElementById("turn-counter").textContent = `Turn ${debateState.turnCount}/20`;

    // Check for derail (AI response contains "DERAIL" anywhere)
    if (fullText.toUpperCase().includes("DERAIL")) {
      debateState.status = "derailed";
      // Replace the streamed DERAIL text with a styled message
      assistantBubble.remove();
      showDerailUI();
    }
    // Check for conclusion
    else if (fullText.includes("I was convinced") || fullText.includes("I was not convinced") || debateState.turnCount >= 20) {
      if (fullText.includes("I was convinced")) {
        debateState.status = "concluded_convinced";
      } else {
        debateState.status = "concluded_not_convinced";
      }
      showConclusionUI(fullText);
    }
  } catch (err) {
    textEl.textContent = "Error: " + err.message;
  }

  debateState.isStreaming = false;
  if (sendBtn) sendBtn.disabled = false;
  if (inputEl) inputEl.disabled = false;
}

function appendChatMessage(role, content) {
  const chatEl = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-${role}`;
  bubble.innerHTML = `<div class="chat-message-text">${content ? renderInlineMarkdown(content) : '<span class="typing-indicator">...</span>'}</div>`;
  chatEl.appendChild(bubble);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function showDerailUI() {
  const inputArea = document.getElementById("chat-input-area");
  if (inputArea) inputArea.innerHTML = `<div class="derail-message">Derail \u{1F682} (You went off-point!)</div>`;
}

function showConclusionUI(verdict) {
  const inputArea = document.getElementById("chat-input-area");
  if (inputArea) inputArea.style.display = "none";
  const conclusionEl = document.getElementById("debate-conclusion");
  if (!conclusionEl) return;
  conclusionEl.style.display = "block";
  conclusionEl.innerHTML = `
    <div class="verdict-text">${escapeHtml(verdict)}</div>
    <div class="verdict-question">Do you agree with this result?</div>
    <div class="verdict-buttons">
      <button class="verdict-btn verdict-agree" onclick="submitDebateFeedback(true)">\u{1F44D}</button>
      <button class="verdict-btn verdict-disagree" onclick="submitDebateFeedback(false)">\u{1F44E}</button>
    </div>`;
}

async function submitDebateFeedback(agrees) {
  try {
    await fetch(
      `${API_BASE_URL}/api/review/${debateState.currentKey}/debate/${debateState.sessionId}/feedback`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_agrees: agrees }),
      }
    );
    const btns = document.querySelector(".verdict-buttons");
    if (btns) btns.innerHTML = `<span style="color:var(--gray-500);">Thank you for your feedback!</span>`;
  } catch {}
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
  // 1. Protect math expressions from being mangled by other replacements
  const mathBlocks = [];
  // Display math: $$...$$
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (m, math) => {
    const idx = mathBlocks.length;
    mathBlocks.push(`<span class="math-display">$$${math}$$</span>`);
    return `%%MATH${idx}%%`;
  });
  // Inline math: $...$  (but not $$)
  text = text.replace(/\$([^\$\n]+?)\$/g, (m, math) => {
    const idx = mathBlocks.length;
    mathBlocks.push(`<span class="math-inline">$${math}$</span>`);
    return `%%MATH${idx}%%`;
  });

  // 2. Protect fenced code blocks: ```lang\n...\n```
  const codeBlocks = [];
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (m, lang, code) => {
    const idx = codeBlocks.length;
    const escaped = escapeHtml(code.trim());
    codeBlocks.push(`<pre class="code-block"><code${lang ? ` class="language-${lang}"` : ""}>${escaped}</code></pre>`);
    return `%%CODE${idx}%%`;
  });

  // 3. Handle citations and links
  let result = text.replace(/\[\[(\d+)\]\]\((#[^)]+)\)/g, (match, num, url) => {
    return `<a href="${url}" class="citation-ref" style="scroll-behavior:smooth;">[${num}]</a>`;
  });
  result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
    if (url.startsWith("#")) {
      return `<a href="${url}" style="scroll-behavior:smooth;">${label}</a>`;
    }
    return `<a href="${url}" target="_blank" rel="noopener">${label}</a>`;
  });
  result = result.replace(/(?<![<\w])(?<!\[)\[(\d+)\](?!\()/g, (match, num) => {
    return `<a href="#ref${num}" class="citation-ref" style="scroll-behavior:smooth;">[${num}]</a>`;
  });

  // 4. Standard markdown formatting
  result = result
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");

  // 5. Restore protected blocks
  result = result.replace(/%%CODE(\d+)%%/g, (m, idx) => codeBlocks[parseInt(idx)]);
  result = result.replace(/%%MATH(\d+)%%/g, (m, idx) => mathBlocks[parseInt(idx)]);

  return result;
}

function typsetMath(el) {
  // Call KaTeX auto-render on an element after content is injected
  if (window.renderMathInElement) {
    window.renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      macros: {
        "\\mathbbm": "\\mathbb",
        "\\bm": "\\boldsymbol",
      },
    });
  }
}

function renderCitation(text) {
  let stripped = text.replace(/^\[\d+\]\s*/, "");

  // Detect and strip [BEFORE]/[AFTER] date tags
  let dateBadge = "";
  const beforeMatch = stripped.match(/\s*\[BEFORE\]\s*$/i);
  const afterMatch = stripped.match(/\s*\[AFTER\]\s*$/i);
  if (beforeMatch) {
    stripped = stripped.replace(/\s*\[BEFORE\]\s*$/i, "");
    dateBadge = '<span class="ref-date-badge ref-before">Published before this paper</span>';
  } else if (afterMatch) {
    stripped = stripped.replace(/\s*\[AFTER\]\s*$/i, "");
    dateBadge = '<span class="ref-date-badge ref-after">Published after this paper</span>';
  }

  const rendered = stripped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return dateBadge + rendered;
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
