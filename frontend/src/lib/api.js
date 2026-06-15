const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * POST /cva
 *
 * Throws an Error with a readable message on non-2xx responses (including
 * FastAPI's 422 validation errors, where `detail` may be a string or a
 * list of pydantic error objects).
 */
export async function computeCva(params) {
  const res = await fetch(`${API_BASE}/cva`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail
          .map((e) => `${e.loc?.slice(-1)[0]}: ${e.msg}`)
          .join("; ");
      }
    } catch {
      // response wasn't JSON -- fall back to statusText
    }
    throw new Error(detail);
  }

  return res.json();
}