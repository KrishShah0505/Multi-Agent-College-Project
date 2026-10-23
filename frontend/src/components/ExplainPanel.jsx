import { useState } from "react";
import { explainCva } from "../lib/agentsApi";
import AgentChart from "./AgentCharts";
import "./explain.css";

const SUGGESTIONS = [
  "Why is the CVA this size?",
  "How risky is the counterparty?",
  "When is my exposure highest?",
];

const STATUS = {
  verified: { label: "Verified", cls: "ok",
    note: "Every number was checked against the engine output, and the wording was reviewed." },
  partially_verified: { label: "Numbers verified", cls: "warn",
    note: "Numbers were checked against the engine output. The wording review was unavailable." },
  fallback: { label: "Standard text", cls: "warn",
    note: "The AI-written text did not pass the checks, so a standard explanation is shown instead." },
  engine_check_failed: { label: "Engine check failed", cls: "bad",
    note: "The engine output failed its own consistency checks, so no explanation was generated." },
};

const clean = (p) => Object.fromEntries(Object.entries(p).filter(([, v]) => v !== null && v !== undefined));

function TraceDetails({ trace, verification }) {
  const issues = verification?.issues ?? [];
  return (
    <details className="trace">
      <summary>How this was checked</summary>
      <div className="trace-stats">
        {trace.attempts.length} attempt{trace.attempts.length === 1 ? "" : "s"} ·{" "}
        {trace.llm_calls ?? "?"} model calls · {trace.tokens ?? "?"} tokens · {trace.seconds}s
      </div>
      {trace.attempts.map((a) => {
        const msgs = [...a.summary_issues, ...a.chart_issues];
        return (
          <div key={a.attempt} className="trace-attempt">
            <strong>Attempt {a.attempt}</strong>{" "}
            <span className="muted">(wording review: {a.review})</span>
            {msgs.length === 0
              ? <div className="trace-ok">No problems found.</div>
              : <ul>{msgs.map((m, i) => <li key={i}><code>{m.code}</code> {m.message}</li>)}</ul>}
          </div>
        );
      })}
      {issues.length > 0 && (
        <div className="trace-attempt">
          <strong>Unresolved</strong>
          <ul>{issues.map((m, i) => <li key={i}><code>{m.code}</code> {m.message}</li>)}</ul>
        </div>
      )}
    </details>
  );
}

export default function ExplainPanel({ params }) {
  const [question, setQuestion] = useState("");
  const [data, setData] = useState(null);
  const [ranWith, setRanWith] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const current = JSON.stringify(clean(params));
  const stale = data && ranWith !== current;

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const out = await explainCva(clean(params), question.trim());
      setData(out);
      setRanWith(current);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const status = data ? STATUS[data.status] ?? { label: data.status, cls: "warn", note: "" } : null;

  return (
    <div className="card explain">
      <div className="chart-title">Plain-language explanation</div>
      <p className="explain-sub">
        AI agents explain this result for non-specialists. A verifier checks every number against the engine.
      </p>

      <div className="explain-form">
        <input
          type="text" value={question} maxLength={200}
          placeholder="What do you want to understand? (optional)"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !loading) run(); }}
        />
        <button className="explain-btn" onClick={run} disabled={loading}>
          {loading ? "Working…" : "Explain this result"}
        </button>
      </div>
      <div className="chips">
        {SUGGESTIONS.map((s) => (
          <button key={s} type="button" className="chip" disabled={loading}
            onClick={() => setQuestion(s)}>{s}</button>
        ))}
      </div>

      {loading && <div className="explain-loading">Agents are working. This can take up to a minute.</div>}
      {error && <div className="error-msg">{error}</div>}

      {data && !loading && (
        <div className="explain-body">
          <div className="status-row">
            <span className={`badge ${status.cls}`}>{status.label}</span>
            <span className="muted">{status.note}</span>
          </div>
          {stale && (
            <div className="stale">Parameters have changed since this explanation was made. Run it again to update.</div>
          )}

          {data.summary ? (
            <>
              <h3 className="explain-headline">{data.summary.headline}</h3>
              {data.summary.explanation.split("\n\n").map((p, i) => <p key={i} className="explain-p">{p}</p>)}

              <div className="key-numbers">
                {data.summary.key_numbers.map((k) => (
                  <div key={k.fact_key} className="key-number">
                    <div className="stat-label">{k.label}</div>
                    <div className="stat-value">{k.value}</div>
                  </div>
                ))}
              </div>

              <ul className="caveats">
                {data.summary.caveats.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </>
          ) : (
            <ul className="caveats">
              {data.verification.issues.map((m, i) => <li key={i}>{m.message}</li>)}
            </ul>
          )}

          {data.charts.map((c) => <AgentChart key={c.id} chart={c} />)}
          <TraceDetails trace={data.trace} verification={data.verification} />
        </div>
      )}
    </div>
  );
}
