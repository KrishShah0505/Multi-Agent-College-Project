import { useState } from "react";

function Field({ label, suffix, value, onChange, ...rest }) {
  // Store raw string so user can type "100000" without cursor jumping.
  // Only commit the parsed number on blur.
  const [raw, setRaw] = useState(String(value));
  const [focused, setFocused] = useState(false);

  const displayed = focused ? raw : String(value);

  return (
    <div className="field">
      <label>{label}</label>
      <div className="field-wrap">
        <input
          {...rest}
          type="text"
          inputMode="decimal"
          value={displayed}
          onChange={(e) => setRaw(e.target.value)}
          onFocus={() => { setFocused(true); setRaw(String(value)); }}
          onBlur={() => {
            setFocused(false);
            const parsed = parseFloat(raw);
            if (!isNaN(parsed)) onChange(parsed);
            else setRaw(String(value)); // reset if garbage
          }}
        />
        {suffix && <span className="field-suffix">{suffix}</span>}
      </div>
    </div>
  );
}

function Section({ title, children, single }) {
  return (
    <fieldset className="form-section">
      <legend>{title}</legend>
      <div className={`form-grid${single ? " single" : ""}`}>{children}</div>
    </fieldset>
  );
}

export default function ParametersPanel({ params, setParam, onSubmit, loading, error }) {
  const set = (key) => (val) => setParam(key, val);

  return (
    <form
      className="card params-form"
      onSubmit={(e) => { e.preventDefault(); onSubmit(); }}
    >
      <Section title="Swap terms">
        <Field label="Notional" suffix="USD"
          value={params.notional} onChange={set("notional")} min="1" step="1000" />
        <Field label="Tenor" suffix="years"
          value={params.tenor_years} onChange={set("tenor_years")} min="0.5" step="0.5" />
        <Field label="Reset frequency" suffix="months"
          value={params.payment_freq_months} onChange={set("payment_freq_months")} min="1" step="1" />

        <div className="field" style={{ gridColumn: "1 / -1" }}>
          <label>Fixed rate</label>
          <div className="fixed-rate-row">
            <input
              type="checkbox"
              id="auto-fixed"
              checked={params.fixed_rate === null}
              onChange={(e) => setParam("fixed_rate", e.target.checked ? null : 0.06)}
            />
            <label htmlFor="auto-fixed" className="fixed-rate-label">use par rate</label>
            {params.fixed_rate !== null && (
              <Field
                label=""
                value={params.fixed_rate}
                onChange={set("fixed_rate")}
                step="0.001"
                style={{ marginTop: 0 }}
              />
            )}
          </div>
        </div>
      </Section>

      <Section title="Rate model (Vasicek)">
        <Field label="r0 — initial rate"
          value={params.r0} onChange={set("r0")} step="0.001" />
        <Field label="theta — long-run mean"
          value={params.theta} onChange={set("theta")} step="0.001" />
        <Field label="kappa — mean reversion"
          value={params.kappa} onChange={set("kappa")} step="0.01" min="0.001" />
        <Field label="sigma — volatility"
          value={params.sigma} onChange={set("sigma")} step="0.001" min="0" />
      </Section>

      <Section title="Counterparty credit">
        <Field label="CDS spread" suffix="dec"
          value={params.cds_spread} onChange={set("cds_spread")} step="0.0005" min="0" />
        <Field label="Recovery rate"
          value={params.recovery_rate} onChange={set("recovery_rate")} step="0.05" min="0" max="0.99" />
      </Section>

      <Section title="Simulation">
        <Field label="Paths"
          value={params.n_paths} onChange={set("n_paths")} step="500" min="100" max="50000" />
        <Field label="Seed"
          value={params.seed} onChange={set("seed")} step="1" />
      </Section>

      <button type="submit" className="run-btn" disabled={loading}>
        {loading ? "Running simulation…" : "Run simulation"}
      </button>

      {error && <div className="error-msg">{error}</div>}
    </form>
  );
}