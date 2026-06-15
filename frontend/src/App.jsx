import { useState } from "react";
import ParametersPanel from "./components/ParametersPanel";
import CvaSummary from "./components/CvaSummary";
import ExposureChart from "./components/ExposureChart";
import { computeCva } from "./lib/api";

const DEFAULT_PARAMS = {
  notional: 1_000_000,
  tenor_years: 5,
  payment_freq_months: 6,
  fixed_rate: null,
  kappa: 0.3,
  theta: 0.06,
  sigma: 0.01,
  r0: 0.05,
  cds_spread: 0.015,
  recovery_rate: 0.4,
  n_paths: 5000,
  seed: 42,
  pfe_quantile: 0.95,
};

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const setParam = (key, value) =>
    setParams((prev) => ({ ...prev, [key]: value }));

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = { ...params };
      if (payload.fixed_rate === null) delete payload.fixed_rate;
      const data = await computeCva(payload);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <header className="app-header">
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <h1>CVA Engine</h1>
          <p>
            Monte Carlo Credit Valuation Adjustment for a vanilla fixed-for-floating
            interest rate swap, under a Vasicek short-rate model.
          </p>
        </div>
      </header>

      <main className="app-main">
        <ParametersPanel
          params={params}
          setParam={setParam}
          onSubmit={run}
          loading={loading}
          error={error}
        />
        <div className="right-col">
          <CvaSummary result={result} notional={params.notional} />
          <ExposureChart result={result} />
        </div>
      </main>

      <footer className="app-footer">
        CVA = (1 − R) · Σ EPE(tᵢ) · PD(tᵢ₋₁, tᵢ) &nbsp;|&nbsp; λ = CDS spread / (1 − R)
      </footer>
    </>
  );
}