import {
  ComposedChart, Area, Line,
  XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";

function fmt$(x) { return "$" + Math.round(x).toLocaleString("en-US"); }

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  const epe = payload.find((p) => p.dataKey === "epe")?.value ?? 0;
  const pfe = payload.find((p) => p.dataKey === "pfe")?.value ?? 0;
  return (
    <div className="chart-tooltip">
      <div className="tooltip-time">t = {Number(label).toFixed(2)}y</div>
      <div className="tooltip-row">
        <span className="tooltip-epe">EPE</span>
        <span>{fmt$(epe)}</span>
      </div>
      <div className="tooltip-row">
        <span className="tooltip-pfe">PFE</span>
        <span>{fmt$(pfe)}</span>
      </div>
      <div className="tooltip-row tooltip-gap">
        <span>gap</span>
        <span>{fmt$(pfe - epe)}</span>
      </div>
    </div>
  );
}

export default function ExposureChart({ result }) {
  if (!result) {
    return (
      <div className="card chart-empty">
        Exposure profile will appear here.
      </div>
    );
  }

  const data = result.time_grid.map((t, i) => ({
    t,
    epe: result.epe[i],
    pfe: result.pfe[i],
    gap: Math.max(result.pfe[i] - result.epe[i], 0),
  }));

  const tickStyle = { fill: "#5E6B7A", fontFamily: "IBM Plex Mono, monospace", fontSize: 11 };

  return (
    <div className="card">
      <div className="chart-header">
        <span className="chart-title">Exposure profile</span>
        <div className="chart-legend">
          <span className="legend-item">
            <span className="legend-line" style={{ background: "#2E6F95" }} /> EPE
          </span>
          <span className="legend-item">
            <span className="legend-line" style={{ background: "#C0562F", opacity: 0.7 }} /> PFE 95th
          </span>
          <span className="legend-item">
            <span className="legend-box" style={{ background: "rgba(192,86,47,0.15)" }} /> convexity gap
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#D8DEE5" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t" type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(t) => `${t}y`}
            tick={tickStyle} stroke="#D8DEE5"
          />
          <YAxis
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            tick={tickStyle} stroke="#D8DEE5" width={54}
          />
          <Tooltip content={<ChartTooltip />} />

          {/* stacked areas: EPE base (invisible) + gap on top */}
          <Area dataKey="epe" stackId="s" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area dataKey="gap" stackId="s" stroke="none" fill="#C0562F" fillOpacity={0.12} isAnimationActive={false} />

          <Line dataKey="pfe" stroke="#C0562F" strokeWidth={1.5} strokeDasharray="4 3"
            dot={{ r: 2.5, fill: "#C0562F", strokeWidth: 0 }} isAnimationActive={false} />
          <Line dataKey="epe" stroke="#2E6F95" strokeWidth={2}
            dot={{ r: 2.5, fill: "#2E6F95", strokeWidth: 0 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}