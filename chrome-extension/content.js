// Content script : extrait les commentaires visibles sur YouTube

function extractComments() {
  const selectors = [
    "#content-text",
    "yt-formatted-string.style-scope.ytd-comment-renderer",
    "ytd-comment-renderer #content-text",
  ];

  const texts = new Set();
  for (const sel of selectors) {
    document.querySelectorAll(sel).forEach(el => {
      const t = el.innerText?.trim();
      if (t && t.length > 3) texts.add(t);
    });
  }
  return [...texts];
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.action === "getComments") {
    sendResponse({ comments: extractComments() });
  }
  return true;
});
