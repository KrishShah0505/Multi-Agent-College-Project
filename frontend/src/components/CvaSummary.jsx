function fmt$(x) { return "$" + Math.round(x).toLocaleString("en-US"); }
function fmtPct(x, d = 4) { return (x * 100).toFixed(d) + "%"; }

function Stat({ label, value }) {
  return (
    <div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

export default function CvaSummary({ result, notional }) {
  if (!result) {
    return (
      <div className="card cva-empty">
        Set parameters and run the simulation to compute CVA.
      </div>
    );
  }

  const { cva, fixed_rate_used, par_rate, hazard } = result;
  const pctNotional = notional > 0 ? cva / notional : 0;

  return (
    <div className="card">
      <div className="cva-top">
        <div>
          <div className="cva-label">Credit valuation adjustment</div>
          <div className="cva-number">{fmt$(cva)}</div>
        </div>
        <div className="cva-pct">{fmtPct(pctNotional, 4)} of notional</div>
      </div>
      <div className="cva-stats">
        <Stat label="Fixed rate used"    value={fmtPct(fixed_rate_used, 4)} />
        <Stat label="Par rate (t = 0)"   value={fmtPct(par_rate, 4)} />
        <Stat label="Implied hazard λ"   value={fmtPct(hazard, 4) + " /yr"} />
      </div>
    </div>
  );
}