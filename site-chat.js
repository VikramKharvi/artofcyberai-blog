const PAGE_URLS = [
  "blog.html",
  "tech-llm-system-map.html",
  "tech-tokenization-model-design.html",
  "tech-transformer-architecture.html",
  "tech-mixture-of-experts.html",
  "tech-build-gpt.html",
  "tech-reproduce-gpt2.html",
  "tech-chatgpt-pipeline.html",
  "post-evals-rl-environments.html",
  "post-edge-of-capability.html",
  "post-graders-shape-models.html",
  "post-real-users-high-entropy.html",
  "post-long-horizon-context.html",
  "post-continual-learning.html",
  "post-taste-bottleneck.html",
  "post-ai-software-engineering.html",
  "post-good-agent-trajectory.html",
  "tech-model-serving-inference.html",
  "inference-service-contract.html",
  "inference-request-path.html",
  "inference-prefill-decode.html",
  "inference-kv-cache.html",
  "inference-continuous-batching.html",
  "inference-parallelism.html",
  "inference-quantization.html",
  "inference-speculative-decoding.html",
  "inference-capacity-reliability.html",
  "inference-load-testing.html"
];

const STOP_WORDS = new Set([
  "about", "after", "also", "and", "are", "because", "been", "before", "being",
  "between", "both", "but", "can", "does", "each", "for", "from", "have", "how",
  "into", "its", "more", "most", "not", "only", "other", "our", "should", "than",
  "that", "the", "their", "then", "there", "these", "they", "this", "through", "use",
  "using", "was", "what", "when", "where", "which", "while", "with", "would", "you"
]);

const state = {
  chunksPromise: null,
  enginePromise: null,
  panel: null,
  messages: null,
  status: null,
  form: null,
  input: null,
  download: null,
  ready: false
};

function normalize(text) {
  return text.replace(/\s+/g, " ").trim();
}

function tokens(text) {
  return (text.toLowerCase().match(/[a-z0-9][a-z0-9+.-]{1,}/g) || [])
    .filter((token) => !STOP_WORDS.has(token));
}

function chunkDocument(html, url) {
  const documentCopy = new DOMParser().parseFromString(html, "text/html");
  const title = normalize(documentCopy.querySelector("h1")?.textContent || documentCopy.title || url);
  const main = documentCopy.querySelector("main");
  if (!main) return [];

  const chunks = [];
  let heading = title;
  let buffer = [];
  let length = 0;

  const flush = () => {
    const text = normalize(buffer.join(" "));
    if (text.length >= 80) chunks.push({ url, title, heading, text });
    buffer = [];
    length = 0;
  };

  main.querySelectorAll("h1, h2, h3, p, li, blockquote, figcaption").forEach((node) => {
    const text = normalize(node.textContent || "");
    if (!text) return;
    if (/^H[1-3]$/.test(node.tagName)) {
      flush();
      heading = text;
      return;
    }
    if (length + text.length > 1050) flush();
    buffer.push(text);
    length += text.length + 1;
  });
  flush();
  return chunks;
}

async function buildIndex() {
  if (state.chunksPromise) return state.chunksPromise;
  state.chunksPromise = Promise.all(PAGE_URLS.map(async (url, index) => {
    setStatus(`Indexing field notes ${index + 1}/${PAGE_URLS.length}...`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not read ${url}`);
    return chunkDocument(await response.text(), url);
  })).then((documents) => documents.flat());
  return state.chunksPromise;
}

function countMatches(text, token) {
  let count = 0;
  let position = text.indexOf(token);
  while (position !== -1) {
    count += 1;
    position = text.indexOf(token, position + token.length);
  }
  return count;
}

function retrieve(chunks, question) {
  const query = question.toLowerCase();
  const queryTokens = [...new Set(tokens(question))];
  const scored = chunks.map((chunk) => {
    const heading = `${chunk.title} ${chunk.heading}`.toLowerCase();
    const body = chunk.text.toLowerCase();
    let score = query.length > 5 && body.includes(query) ? 18 : 0;
    queryTokens.forEach((token) => {
      score += countMatches(heading, token) * 5;
      score += Math.min(4, countMatches(body, token));
    });
    return { ...chunk, score };
  }).filter((chunk) => chunk.score > 0).sort((a, b) => b.score - a.score);

  const selected = [];
  const pageCounts = new Map();
  for (const chunk of scored) {
    const count = pageCounts.get(chunk.url) || 0;
    if (count >= 2) continue;
    selected.push(chunk);
    pageCounts.set(chunk.url, count + 1);
    if (selected.length === 6) break;
  }
  return selected;
}

function setStatus(text) {
  if (state.status) state.status.textContent = text;
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `site-chat-message is-${role}`;
  message.textContent = text;
  state.messages.append(message);
  state.messages.scrollTop = state.messages.scrollHeight;
  return message;
}

function showSources(results) {
  const sources = document.createElement("div");
  sources.className = "site-chat-sources";
  const label = document.createElement("span");
  label.textContent = "Sources";
  sources.append(label);
  [...new Map(results.map((result) => [result.url, result])).values()].forEach((result, index) => {
    const link = document.createElement("a");
    link.href = result.url;
    link.textContent = `${index + 1}. ${result.title}`;
    sources.append(link);
  });
  state.messages.append(sources);
  state.messages.scrollTop = state.messages.scrollHeight;
}

async function createEngine() {
  if (state.enginePromise) return state.enginePromise;
  state.enginePromise = (async () => {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
    setStatus("Connecting to the browser-local model...");
    const webllm = await import("https://esm.run/@mlc-ai/web-llm");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No compatible WebGPU adapter was found.");
    const supportsF16 = adapter.features.has("shader-f16");
    const model = supportsF16
      ? "SmolLM2-360M-Instruct-q4f16_1-MLC"
      : "SmolLM2-360M-Instruct-q4f32_1-MLC";
    const worker = new Worker(new URL("./site-chat-worker.js", import.meta.url), { type: "module" });
    return webllm.CreateWebWorkerMLCEngine(worker, model, {
      initProgressCallback: (progress) => setStatus(progress.text || "Loading the local model...")
    });
  })();
  return state.enginePromise;
}

function fallbackAnswer(results, error) {
  const introduction = error?.message?.includes("WebGPU")
    ? "This browser cannot run the local model, but these passages are the closest match:"
    : "The local model could not start. These passages are the closest match:";
  const text = results.length
    ? `${introduction}\n\n${results.slice(0, 3).map((result) => `${result.heading}: ${result.text.slice(0, 280)}...`).join("\n\n")}`
    : "I could not find a relevant passage in the published field notes.";
  addMessage("assistant", text);
  if (results.length) showSources(results);
  setStatus("Search mode");
}

async function answer(question) {
  if (!state.ready) return;
  state.form.querySelector("button").disabled = true;
  addMessage("user", question);
  try {
    const chunks = await buildIndex();
    const results = retrieve(chunks, question);
    if (!results.length) {
      fallbackAnswer([], null);
      return;
    }

    let engine;
    try {
      engine = await createEngine();
    } catch (error) {
      fallbackAnswer(results, error);
      return;
    }

    setStatus("Answering locally with WebLLM...");
    const context = results.map((result, index) =>
      `[${index + 1}] ${result.title} — ${result.heading}\nURL: ${result.url}\n${result.text}`
    ).join("\n\n");
    const responseNode = addMessage("assistant", "");
    const stream = await engine.chat.completions.create({
      messages: [
        {
          role: "system",
          content: "Answer only from the supplied website excerpts. Be concise and technical. Cite supporting excerpts with bracketed numbers such as [1]. If the excerpts do not answer the question, say so clearly."
        },
        {
          role: "user",
          content: `Website excerpts:\n\n${context}\n\nQuestion: ${question}`
        }
      ],
      temperature: 0.2,
      max_tokens: 360,
      stream: true
    });
    let answerText = "";
    for await (const chunk of stream) {
      answerText += chunk.choices[0]?.delta?.content || "";
      responseNode.textContent = answerText;
      state.messages.scrollTop = state.messages.scrollHeight;
    }
    showSources(results);
    setStatus("Ready — model and pages stay in your browser");
  } catch (error) {
    addMessage("assistant", `I could not answer that question: ${error.message}`);
    setStatus("Something went wrong");
  } finally {
    state.form.querySelector("button").disabled = false;
    state.input.focus();
  }
}

async function prepareChat() {
  state.download.disabled = true;
  state.download.querySelector("strong").textContent = "Preparing local AI...";
  try {
    await buildIndex();
    await createEngine();
    state.ready = true;
    state.download.hidden = true;
    state.input.disabled = false;
    state.form.querySelector("button").disabled = false;
    addMessage("assistant", "The local model is ready. Ask a question about any published chapter.");
    setStatus("Ready — inference runs in your browser");
    state.input.focus();
  } catch (error) {
    state.enginePromise = null;
    state.download.disabled = false;
    state.download.querySelector("strong").textContent = "Try downloading again";
    addMessage("assistant", `The local model could not start: ${error.message}`);
    setStatus("A WebGPU-capable browser is required");
  }
}

function openPanel() {
  state.panel.hidden = false;
  requestAnimationFrame(() => state.panel.classList.add("is-open"));
  state.input.focus();
}

function closePanel() {
  state.panel.classList.remove("is-open");
  window.setTimeout(() => { state.panel.hidden = true; }, 180);
}

function initializeChat() {
  const launcher = document.createElement("button");
  launcher.className = "site-chat-launcher";
  launcher.type = "button";
  launcher.setAttribute("aria-haspopup", "dialog");
  launcher.innerHTML = "<span>Ask the field notes</span><b aria-hidden=\"true\">AI</b>";

  const panel = document.createElement("aside");
  panel.className = "site-chat-panel";
  panel.hidden = true;
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", "false");
  panel.setAttribute("aria-label", "Ask the field notes");
  panel.innerHTML = `
    <header class="site-chat-header">
      <div><small>Browser-local research assistant</small><h2>Ask the field notes</h2></div>
      <button type="button" aria-label="Close chat">&times;</button>
    </header>
    <div class="site-chat-messages" aria-live="polite">
      <div class="site-chat-message is-assistant">Ask about pre-training, post-training, RL environments, agent trajectories, or model serving. Nothing downloads until you choose to start the local model.</div>
    </div>
    <button class="site-chat-download" type="button"><strong>Download local AI</strong><span>First use &middot; approximately 400&ndash;600 MB</span></button>
    <form class="site-chat-form">
      <label for="site-chat-question">Your question</label>
      <div><input id="site-chat-question" name="question" autocomplete="off" placeholder="How should an RL trajectory be recorded?" required disabled><button type="submit" disabled>Ask</button></div>
    </form>
    <footer class="site-chat-meta"><span>Local inference</span><span class="site-chat-status">Download required</span><a href="https://github.com/mlc-ai/web-llm" target="_blank" rel="noreferrer">Powered by WebLLM &nearr;</a></footer>`;

  document.body.append(launcher, panel);
  state.panel = panel;
  state.messages = panel.querySelector(".site-chat-messages");
  state.status = panel.querySelector(".site-chat-status");
  state.form = panel.querySelector(".site-chat-form");
  state.input = panel.querySelector("input");
  state.download = panel.querySelector(".site-chat-download");

  launcher.addEventListener("click", openPanel);
  state.download.addEventListener("click", prepareChat);
  panel.querySelector(".site-chat-header button").addEventListener("click", closePanel);
  state.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = normalize(state.input.value);
    if (!question) return;
    state.input.value = "";
    answer(question);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) closePanel();
  });
}

initializeChat();
