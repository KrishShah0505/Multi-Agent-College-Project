const AGENTS_BASE = import.meta.env.VITE_AGENTS_URL || "http://localhost:8001";

/**
 * POST /explain on the agents service.
 * `params` is the same body the engine's /cva takes. `question` is optional
 * and steers what the agents emphasise.
 */
export async function explainCva(params, question) {
  const res = await fetch(`${AGENTS_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params, question: question || null }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail.map((e) => `${e.loc?.slice(-1)[0]}: ${e.msg}`).join("; ");
      }
    } catch {
      // not JSON; keep statusText
    }
    throw new Error(detail);
  }
  return res.json();
}
