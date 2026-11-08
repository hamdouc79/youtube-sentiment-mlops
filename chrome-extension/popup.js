const DEFAULT_API = "https://literate-happiness-v6q7j4p5w7qg3xvr5-7860.app.github.dev";

const $ = id => document.getElementById(id);

let allResults = [];
let activeFilter = "all";

// --- Persist API URL and dark mode ---
chrome.storage.local.get(["apiUrl", "dark"], ({ apiUrl, dark }) => {
  $("apiUrl").value = apiUrl || DEFAULT_API;
  if (dark) document.body.classList.add("dark");
});

$("toggleDark").addEventListener("click", () => {
  const on = document.body.classList.toggle("dark");
  chrome.storage.local.set({ dark: on });
});

$("analyzeBtn").addEventListener("click", runAnalysis);

async function runAnalysis() {
  const apiUrl = $("apiUrl").value.trim().replace(/\/$/, "");
  chrome.storage.local.set({ apiUrl });

  showStatus("Extraction des commentaires...");

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab.url?.includes("youtube.com/watch")) {
    showStatus("Ouvre une vidéo YouTube d'abord.", true);
    return;
  }

  let comments = [];
  try {
    const res = await chrome.tabs.sendMessage(tab.id, { action: "getComments" });
    comments = res?.comments ?? [];
  } catch {
    showStatus("Impossible d'accéder à la page. Recharge la page YouTube.", true);
    return;
  }

  if (comments.length === 0) {
    showStatus("Aucun commentaire trouvé. Fais défiler la page d'abord.", true);
    return;
  }

  showStatus(`Analyse de ${comments.length} commentaires...`);

  try {
    const resp = await fetch(`${apiUrl}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });
    if (!resp.ok) throw new Error(`API erreur ${resp.status}`);
    const data = await resp.json();
    renderResults(data, comments.length);
  } catch (e) {
    showStatus(`Erreur API : ${e.message}`, true);
  }
}

function renderResults(data, total) {
  allResults = data.results;
  hideStatus();

  // Stats
  const s = data.stats;
  const pos = s["positif"]?.pct ?? 0;
  const neu = s["neutre"]?.pct ?? 0;
  const neg = s["négatif"]?.pct ?? (s["négatif"]?.pct ?? 0);

  $("stat-pos").querySelector("span").textContent = pos;
  $("stat-neu").querySelector("span").textContent = neu;
  $("stat-neg").querySelector("span").textContent = neg;
  $("totalCount").textContent = `${total} commentaires analysés en ${data.inference_ms}ms`;

  $("stats").classList.remove("hidden");
  $("filters").classList.remove("hidden");
  renderList();
}

function renderList() {
  const list = $("commentList");
  list.innerHTML = "";
  const filtered = activeFilter === "all"
    ? allResults
    : allResults.filter(r => r.sentiment === activeFilter);

  if (filtered.length === 0) {
    list.innerHTML = '<li style="color:var(--text2);font-size:12px;padding:6px">Aucun commentaire pour ce filtre.</li>';
  } else {
    filtered.forEach(r => {
      const li = document.createElement("li");
      li.className = "comment-item";
      li.dataset.sentiment = r.sentiment.startsWith("pos") ? "positif"
        : r.sentiment.startsWith("nég") || r.sentiment.startsWith("neg") ? "negatif"
        : "neutre";
      const emoji = li.dataset.sentiment === "positif" ? "😊"
        : li.dataset.sentiment === "negatif" ? "😠" : "😐";
      li.innerHTML = `
        <div>${escapeHtml(r.comment)}</div>
        <div class="comment-meta">${emoji} ${r.sentiment} · confiance ${(r.confidence * 100).toFixed(1)}%</div>
      `;
      list.appendChild(li);
    });
  }
  list.classList.remove("hidden");
}

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activeFilter = btn.dataset.filter;
    renderList();
  });
});

$("copyBtn").addEventListener("click", () => {
  const text = allResults.map(r => `[${r.sentiment}] ${r.comment}`).join("\n");
  navigator.clipboard.writeText(text).then(() => {
    $("copyBtn").textContent = "✅ Copié !";
    setTimeout(() => { $("copyBtn").textContent = "📋 Copier"; }, 2000);
  });
});

function showStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.remove("hidden", "error");
  if (isError) el.classList.add("error");
}
function hideStatus() { $("status").classList.add("hidden"); }

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
