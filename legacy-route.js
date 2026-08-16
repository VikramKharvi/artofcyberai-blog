(() => {
  const host = window.location.hostname.replace(/^www\./, "");
  if (host !== "trainrl.com") return;

  const path = window.location.pathname.split("/").pop() || "index.html";
  const hash = window.location.hash;
  const search = window.location.search;
  const preBase = "https://pre-trained.artofcyberai.com/";
  const inferenceBase = "https://inference.artofcyberai.com/";
  const preRoutes = new Set([
    "tech-llm-system-map.html",
    "tech-tokenization-model-design.html",
    "tech-transformer-architecture.html",
    "tech-mixture-of-experts.html",
  ]);
  const inferenceRoutes = new Set([
    "inference-service-contract.html",
    "inference-request-path.html",
    "inference-prefill-decode.html",
    "inference-kv-cache.html",
    "inference-continuous-batching.html",
    "inference-parallelism.html",
    "inference-quantization.html",
    "inference-speculative-decoding.html",
    "inference-capacity-reliability.html",
    "inference-load-testing.html",
  ]);

  let target = null;
  let preserveHash = true;

  if (path === "blog.html") {
    preserveHash = false;
    target = hash === "#tech-notes"
      ? preBase
      : hash === "#serving"
        ? inferenceBase
        : hash === "#post-training"
          ? "https://trainrl.com/"
          : "https://trainrl.com/";
  } else if (preRoutes.has(path)) {
    target = preBase + path;
  } else if (inferenceRoutes.has(path)) {
    target = inferenceBase + path;
  } else if (path === "tech-build-gpt.html") {
    preserveHash = false;
    target = preBase + "tech-tokenization-model-design.html#attention";
  } else if (path === "tech-reproduce-gpt2.html") {
    preserveHash = false;
    target = preBase + "tech-tokenization-model-design.html#training";
  } else if (path === "tech-model-serving-inference.html") {
    preserveHash = false;
    const servingAnchors = {
      "#contract": "inference-service-contract.html",
      "#request-path": "inference-request-path.html",
      "#prefill-decode": "inference-prefill-decode.html",
      "#kv-cache": "inference-kv-cache.html",
      "#scheduling": "inference-continuous-batching.html",
      "#parallelism": "inference-parallelism.html",
      "#precision": "inference-quantization.html",
      "#acceleration": "inference-speculative-decoding.html",
      "#reliability": "inference-capacity-reliability.html",
      "#evidence": "inference-load-testing.html",
    };
    target = inferenceBase + (servingAnchors[hash] || "inference-service-contract.html");
  }

  if (target) {
    window.location.replace(target + search + (preserveHash ? hash : ""));
  }
})();
