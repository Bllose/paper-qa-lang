/**
 * PaperQA Chat — 前端应用
 *
 * 扩展点（用于后续功能）：
 *   - this.#onMessageComplete    : 每条消息完成时的回调
 *   - this.#onTokenReceived      : 每个 token 到达时的回调
 *   - this.#papersView           : 论文列表视图（占位）
 *   - this.updateStats()         : 外部可调用的统计更新
 */
class ChatApp {
  static #instance = null;

  static getInstance() {
    if (!ChatApp.#instance) ChatApp.#instance = new ChatApp();
    return ChatApp.#instance;
  }

  // ── 扩展回调（外部可注入） ──
  onMessageComplete = null;   // (role, content) => void
  onTokenReceived = null;     // (token, totalTokens) => void

  // ── 状态 ──
  #abortController = null;
  #healthInterval = null;
  #messages = [];        // { role, content }
  #totalTokens = 0;
  #isStreaming = false;
  #resizeObserver = null;

  // ── DOM 引用 ──
  #el = {};
  #resizeHandlers = [];

  constructor() {
    if (ChatApp.#instance) throw new Error("Use ChatApp.getInstance()");
    this.#init();
  }

  async #init() {
    this.#cacheElements();
    this.#bindEvents();
    this.#startResizeObserver();
    this.#startHealthCheck();
  }

  // ── DOM 缓存 ──

  #cacheElements() {
    this.#el.messages = document.getElementById("messages");
    this.#el.welcome = document.getElementById("welcome");
    this.#el.input = document.getElementById("chat-input");
    this.#el.sendBtn = document.getElementById("send-btn");
    this.#el.tokenCount = document.getElementById("token-count");
    this.#el.msgCount = document.getElementById("msg-count");
    this.#el.winHeight = document.getElementById("win-height");
    this.#el.healthDot = document.getElementById("health-dot");
    this.#el.healthLabel = document.getElementById("health-label");
    this.#el.healthBadge = document.getElementById("health-badge");
  }

  // ── 事件绑定 ──

  #bindEvents() {
    this.#el.sendBtn.addEventListener("click", () => this.#onSend());
    this.#el.input.addEventListener("keydown", (e) => this.#onKeydown(e));
    this.#el.input.addEventListener("input", () => this.#onInput());
  }

  #onKeydown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      this.#onSend();
    }
  }

  #onInput() {
    const ta = this.#el.input;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
    this.#el.sendBtn.disabled = !ta.value.trim();
  }

  // ── 窗口大小监测 ──

  #startResizeObserver() {
    if (!window.ResizeObserver) return;
    this.#resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.target === this.#el.messages) {
          const h = Math.round(entry.contentRect.height);
          this.#el.winHeight.textContent = h;
          // 通知外部注册的 resize 回调
          for (const fn of this.#resizeHandlers) fn({ height: h, width: Math.round(entry.contentRect.width) });
        }
      }
    });
    this.#resizeObserver.observe(this.#el.messages);
    // 初始值
    requestAnimationFrame(() => {
      this.#el.winHeight.textContent = this.#el.messages.clientHeight;
    });
  }

  /** 注册窗口 resize 回调（扩展用） */
  onResize(fn) {
    this.#resizeHandlers.push(fn);
  }

  // ── 健康检查 ──

  async #checkHealth() {
    this.#el.healthDot.className = "health-dot checking";
    this.#el.healthLabel.textContent = "检查中...";
    try {
      const res = await fetch("/health", { signal: AbortSignal.timeout(5000) });
      const ok = res.ok;
      this.#el.healthDot.className = `health-dot ${ok ? "online" : "offline"}`;
      this.#el.healthLabel.textContent = ok ? "正常运行" : "异常";
      this.#el.healthBadge.title = ok ? "后端服务正常运行" : `状态码: ${res.status}`;
    } catch {
      this.#el.healthDot.className = "health-dot offline";
      this.#el.healthLabel.textContent = "无法连接";
      this.#el.healthBadge.title = "无法连接到后端服务";
    }
  }

  #startHealthCheck() {
    this.#checkHealth();
    this.#healthInterval = setInterval(() => this.#checkHealth(), 30000);
  }

  // ── 发送消息 ──

  #onSend() {
    const text = this.#el.input.value.trim();
    if (!text || this.#isStreaming) return;

    // 隐藏欢迎语
    this.#el.welcome.classList.add("hidden");

    // 添加用户消息
    this.#addMessage("user", text);
    this.#messages.push({ role: "user", content: text });

    // 清空输入
    this.#el.input.value = "";
    this.#el.input.style.height = "auto";
    this.#el.sendBtn.disabled = true;

    this.#updateStats();
    this.#streamChat(text);
  }

  async #streamChat(text) {
    this.#isStreaming = true;
    this.#abortController = new AbortController();

    // 创建助手消息占位
    const msgDiv = this.#addMessage("assistant", "", true);
    const bubble = msgDiv.querySelector(".bubble");
    let fullContent = "";

    try {
      const response = await fetch("/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: this.#abortController.signal,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const event of events) {
          const data = this.#parseSSEData(event);
          if (data === null) continue;
          if (data === "[DONE]") break;

          try {
            const parsed = typeof data === "string" ? JSON.parse(data) : data;
            if (parsed.token) {
              fullContent += parsed.token;
              this.#totalTokens++;
              this.#renderAssistantContent(bubble, fullContent);
              this.#scrollToBottom();
              this.#updateStats();
              this.onTokenReceived?.(parsed.token, this.#totalTokens);
            } else if (parsed.error) {
              throw new Error(parsed.error);
            }
          } catch (e) {
            if (e.message === "[DONE]") break;
            throw e;
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        // 用户取消
      } else {
        this.#addMessage("error", `出错了: ${err.message}`);
        bubble.textContent = fullContent || "（响应中断）";
      }
    } finally {
      msgDiv.classList.remove("streaming");
      this.#isStreaming = false;
      this.#abortController = null;
      if (fullContent) {
        this.#messages.push({ role: "assistant", content: fullContent });
        this.onMessageComplete?.("assistant", fullContent);
      }
      this.#updateStats();
      this.#scrollToBottom();
      this.#el.input.focus();
    }
  }

  /** 取消当前流式响应 */
  cancelStreaming() {
    if (this.#abortController) {
      this.#abortController.abort();
      this.#abortController = null;
    }
  }

  // ── SSE 解析 ──

  #parseSSEData(event) {
    const lines = event.split("\n").filter((l) => l.startsWith("data: "));
    if (lines.length === 0) return null;
    const data = lines[lines.length - 1].slice(6); // 取最后一行 data
    return data;
  }

  // ── DOM 操作 ──

  #addMessage(role, content, streaming = false) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    if (streaming) div.classList.add("streaming");

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    if (role === "error") {
      bubble.textContent = content;
    } else if (role === "user") {
      bubble.textContent = content;
    }

    div.appendChild(bubble);

    // Meta
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const now = new Date();
    meta.textContent = now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    div.appendChild(meta);

    // 如果是第一条助手消息，隐藏 meta（流式结束后再显示）
    if (streaming) meta.style.display = "none";

    this.#el.messages.appendChild(div);
    this.#scrollToBottom();
    return div;
  }

  #renderAssistantContent(bubble, markdown) {
    if (!markdown) {
      bubble.innerHTML = "";
      return;
    }
    try {
      bubble.innerHTML = marked.parse(markdown, { breaks: true });
    } catch {
      bubble.textContent = markdown;
    }
  }

  #scrollToBottom() {
    requestAnimationFrame(() => {
      this.#el.messages.scrollTop = this.#el.messages.scrollHeight;
    });
  }

  #updateStats() {
    this.#el.tokenCount.textContent = this.#totalTokens;
    this.#el.msgCount.textContent = this.#messages.length;
  }

  /** 外部重置对话（后续功能） */
  reset() {
    this.cancelStreaming();
    this.#messages = [];
    this.#totalTokens = 0;
    this.#el.messages.querySelectorAll(".message").forEach((el) => el.remove());
    this.#el.welcome.classList.remove("hidden");
    this.#updateStats();
  }
}

// ════════════════════════════════════════════════════════════════════
// PapersPanel — 论文列表下拉 + 添加 / 查看
// ════════════════════════════════════════════════════════════════════

class PapersPanel {
  static #instance = null;

  static getInstance() {
    if (!PapersPanel.#instance) PapersPanel.#instance = new PapersPanel();
    return PapersPanel.#instance;
  }

  #paperCache = [];  // { doc_id, title, ... } 当前论文列表

  constructor() {
    if (PapersPanel.#instance) throw new Error("Use PapersPanel.getInstance()");
    this.#init();
  }

  #init() {
    this.#cacheElements();
    this.#bindEvents();
  }

  // ── DOM ──

  #cacheElements() {
    this.el = {
      navBtn:   document.getElementById("papers-nav-btn"),
      dropdown: document.getElementById("papers-dropdown"),
      arrow:    document.getElementById("dropdown-arrow"),
      paperList: document.getElementById("paper-list"),
      addBtn:   document.getElementById("add-paper-btn"),

      // Add modal
      addModal:     document.getElementById("add-paper-modal"),
      addClose:     document.getElementById("add-paper-close"),
      doiInput:     document.getElementById("doi-input"),
      doiSubmitBtn: document.getElementById("doi-submit-btn"),
      uploadZone:   document.getElementById("upload-zone"),
      fileInput:    document.getElementById("file-input"),
      uploadStatus: document.getElementById("upload-status"),
      ingestStatus: document.getElementById("ingest-status"),

      // Detail modal
      detailModal:  document.getElementById("paper-detail-modal"),
      detailClose:  document.getElementById("detail-close"),
      detailTitle:  document.getElementById("detail-title"),
      detailBody:   document.getElementById("detail-body"),
    };
  }

  // ── Events ──

  #bindEvents() {
    // Toggle dropdown
    this.el.navBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.#toggleDropdown();
    });

    // Add paper modal
    this.el.addBtn.addEventListener("click", () => this.#openAddModal());
    this.el.addClose.addEventListener("click", () => this.#closeAddModal());
    this.el.addModal.addEventListener("click", (e) => {
      if (e.target === this.el.addModal) this.#closeAddModal();
    });

    // DOI submit
    this.el.doiSubmitBtn.addEventListener("click", () => this.#ingestDoi());
    this.el.doiInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.#ingestDoi();
    });

    // File upload
    this.el.uploadZone.addEventListener("click", () => this.el.fileInput.click());
    this.el.fileInput.addEventListener("change", () => this.#handleFileSelect());
    this.el.uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.el.uploadZone.classList.add("dragover");
    });
    this.el.uploadZone.addEventListener("dragleave", () => {
      this.el.uploadZone.classList.remove("dragover");
    });
    this.el.uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.el.uploadZone.classList.remove("dragover");
      const files = e.dataTransfer?.files;
      if (files?.length) this.#uploadFile(files[0]);
    });

    // Detail modal
    this.el.detailClose.addEventListener("click", () => this.#closeDetailModal());
    this.el.detailModal.addEventListener("click", (e) => {
      if (e.target === this.el.detailModal) this.#closeDetailModal();
    });
  }

  // ── Dropdown ──

  #dropdownOpen = false;

  #toggleDropdown() {
    if (this.#dropdownOpen) {
      this.#closeDropdown();
    } else {
      this.#dropdownOpen = true;
      this.el.dropdown.style.display = "block";
      this.el.arrow.classList.add("rotated");
      this.el.navBtn.classList.add("active");
      this.#loadPapers();
    }
  }

  #closeDropdown() {
    this.#dropdownOpen = false;
    this.el.dropdown.style.display = "none";
    this.el.arrow.classList.remove("rotated");
    this.el.navBtn.classList.remove("active");
  }

  // ── Paper list ──

  async #loadPapers() {
    try {
      const res = await fetch("/v1/papers");
      if (!res.ok) throw new Error(await res.text());
      const papers = await res.json();
      this.#paperCache = papers;
      this.#renderPaperList();
    } catch (err) {
      this.el.paperList.innerHTML = `<p class="paper-list-error">加载失败: ${err.message}</p>`;
    }
  }

  #renderPaperList() {
    const list = this.el.paperList;
    if (!this.#paperCache.length) {
      list.innerHTML = '<p class="paper-list-empty">暂无论文，点击上方添加</p>';
      return;
    }
    list.innerHTML = this.#paperCache.map((p) => {
      const title = p.title || "无标题";
      const authors = (p.authors || []).slice(0, 2).join(", ");
      const year = p.year || "";
      return `
        <button class="paper-item" data-doc-id="${p.doc_id}">
          <span class="paper-item-title">${this.#escapeHtml(title)}</span>
          <span class="paper-item-meta">${this.#escapeHtml(authors)}${year ? ` · ${year}` : ""}</span>
        </button>`;
    }).join("");

    // Bind click on each paper item
    list.querySelectorAll(".paper-item").forEach((el) => {
      el.addEventListener("click", () => this.#openDetail(el.dataset.docId));
    });
  }

  #escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ── Add modal ──

  #openAddModal() {
    this.el.addModal.style.display = "flex";
    this.el.doiInput.value = "";
    this.el.uploadStatus.textContent = "";
    this.el.ingestStatus.textContent = "";
    this.el.fileInput.value = "";
    this.el.doiInput.focus();
  }

  #closeAddModal() {
    this.el.addModal.style.display = "none";
  }

  // ── DOI ingest ──

  async #ingestDoi() {
    const doi = this.el.doiInput.value.trim();
    if (!doi) return;

    this.el.ingestStatus.textContent = "正在添加...";
    this.el.doiSubmitBtn.disabled = true;
    try {
      const res = await fetch("/v1/papers/ingest/doi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doi }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      this.el.ingestStatus.textContent = `添加成功 (${data.chunks} 个片段)`;
      this.#refreshAfterIngest();
    } catch (err) {
      this.el.ingestStatus.textContent = `添加失败: ${err.message}`;
    } finally {
      this.el.doiSubmitBtn.disabled = false;
    }
  }

  // ── File upload ──

  #handleFileSelect() {
    const file = this.el.fileInput.files?.[0];
    if (file) this.#uploadFile(file);
  }

  async #uploadFile(file) {
    if (!file.name.endsWith(".pdf")) {
      this.el.uploadStatus.textContent = "仅支持 PDF 文件";
      return;
    }
    this.el.uploadStatus.textContent = `正在上传: ${file.name}...`;
    this.el.ingestStatus.textContent = "";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/v1/papers/ingest/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      this.el.uploadStatus.textContent = `上传成功 (${data.chunks} 个片段)`;
      this.#refreshAfterIngest();
    } catch (err) {
      this.el.uploadStatus.textContent = `上传失败: ${err.message}`;
    }
  }

  async #refreshAfterIngest() {
    // Reload paper list and close modal shortly
    await this.#loadPapers();
    setTimeout(() => {
      if (this.el.ingestStatus.textContent.startsWith("添加成功") ||
          this.el.uploadStatus.textContent.startsWith("上传成功")) {
        this.#closeAddModal();
      }
    }, 1200);
  }

  // ── Detail modal ──

  async #openDetail(docId) {
    this.el.detailModal.style.display = "flex";
    this.el.detailTitle.textContent = "加载中...";
    this.el.detailBody.innerHTML = '<p class="detail-loading">加载论文信息...</p>';

    try {
      const res = await fetch(`/v1/papers/${encodeURIComponent(docId)}`);
      if (!res.ok) throw new Error((await res.json()).detail || "Not found");
      const paper = await res.json();
      this.#renderDetail(paper);
    } catch (err) {
      this.el.detailTitle.textContent = "错误";
      this.el.detailBody.innerHTML = `<p class="detail-error">加载失败: ${err.message}</p>`;
    }
  }

  #detailDocId = null;  // 当前详情弹窗对应的 doc_id

  #renderDetail(paper) {
    this.#detailDocId = paper.doc_id;
    this.el.detailTitle.textContent = paper.title || "论文详情";

    const fields = [
      ["DOI", paper.doi],
      ["作者", (paper.authors || []).join("; ")],
      ["年份", paper.year],
      ["期刊", paper.journal],
      ["引用数", paper.citation_count],
      ["来源质量", paper.source_quality],
      ["已撤稿", paper.is_retracted === null ? null : (paper.is_retracted ? "是" : "否")],
      ["出版日期", paper.publication_date],
      ["PDF 链接", paper.pdf_url ? `<a href="${this.#escapeHtml(paper.pdf_url)}" target="_blank" rel="noopener">${this.#escapeHtml(paper.pdf_url)}</a>` : null],
      ["文件路径", paper.file_location],
      ["摘要", paper.abstract],
      ["BibTeX", paper.bibtex],
    ];

    const rows = fields
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([label, value]) => {
        const display = String(value);
        if (label === "BibTeX") {
          return `<div class="detail-field">
            <span class="detail-label">${label}</span>
            <pre class="detail-bibtex">${this.#escapeHtml(display)}</pre>
          </div>`;
        }
        if (label === "摘要" && display.length > 300) {
          return `<div class="detail-field">
            <span class="detail-label">${label}</span>
            <p class="detail-value detail-abstract">${this.#escapeHtml(display)}</p>
          </div>`;
        }
        return `<div class="detail-field">
          <span class="detail-label">${label}</span>
          <span class="detail-value">${value}</span>
        </div>`;
      })
      .join("");

    this.el.detailBody.innerHTML = rows || "<p>无可用信息</p>";

    // 添加删除按钮
    const btnRow = document.createElement("div");
    btnRow.className = "detail-actions";
    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-danger";
    delBtn.textContent = "删除此论文";
    delBtn.addEventListener("click", () => this.#deletePaper());
    btnRow.appendChild(delBtn);
    this.el.detailBody.appendChild(btnRow);
  }

  async #deletePaper() {
    if (!this.#detailDocId) return;
    if (!confirm("确定要删除这篇论文及其全部索引数据吗？此操作不可撤销。")) return;

    try {
      const res = await fetch(`/v1/papers/${encodeURIComponent(this.#detailDocId)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      this.#closeDetailModal();
      this.#loadPapers();
    } catch (err) {
      alert(`删除失败: ${err.message}`);
    }
  }

  #closeDetailModal() {
    this.el.detailModal.style.display = "none";
    this.#detailDocId = null;
  }
}

// ── 启动 ──
document.addEventListener("DOMContentLoaded", () => {
  ChatApp.getInstance();
  PapersPanel.getInstance();
});
