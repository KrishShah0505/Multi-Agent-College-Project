import {
  ComposedChart, Bar, Line,
  XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const COLORS = { epe: "#2E6F95", pfe: "#C0562F", cva: "#B8862B", ink: "#1B2733" };
const tick = { fill: "#5E6B7A", fontFamily: "IBM Plex Mono, monospace", fontSize: 11 };
const margin = { top: 8, right: 16, bottom: 0, left: 0 };

const usd = (v) => "$" + Math.round(v).toLocaleString("en-US");
const usdK = (v) => `$${(v / 1000).toFixed(0)}k`;
const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;
const yrs = (t) => `${t}y`;

const label = (chart, key) => chart.series.find((s) => s.key === key)?.label ?? key;

function Frame({ chart, children }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={chart.data} margin={margin}>
        <CartesianGrid stroke="#D8DEE5" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="t" tickFormatter={yrs} tick={tick} stroke="#D8DEE5" />
        {children}
        <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 11 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function ExposureProfile({ chart }) {
  return (
    <Frame chart={chart}>
      <YAxis tickFormatter={usdK} tick={tick} stroke="#D8DEE5" width={54} />
      <Tooltip formatter={(v) => usd(v)} labelFormatter={(t) => `Year ${t}`} />
      <Line dataKey="pfe" name={label(chart, "pfe")} stroke={COLORS.pfe} strokeWidth={1.5}
        strokeDasharray="4 3" dot={{ r: 2.5 }} isAnimationActive={false} />
      <Line dataKey="epe" name={label(chart, "epe")} stroke={COLORS.epe} strokeWidth={2}
        dot={{ r: 2.5 }} isAnimationActive={false} />
    </Frame>
  );
}

function DefaultProbability({ chart }) {
  return (
    <Frame chart={chart}>
      <YAxis tickFormatter={(v) => pct(v, 0)} tick={tick} stroke="#D8DEE5" width={44} />
      <Tooltip formatter={(v) => pct(v)} labelFormatter={(t) => `Period ending year ${t}`} />
      <Bar dataKey="marginal_pd" name={label(chart, "marginal_pd")} fill={COLORS.pfe}
        fillOpacity={0.6} isAnimationActive={false} />
      <Line dataKey="cumulative_pd" name={label(chart, "cumulative_pd")} stroke={COLORS.ink}
        strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
    </Frame>
  );
}

function CvaContribution({ chart }) {
  return (
    <Frame chart={chart}>
      <YAxis yAxisId="left" tickFormatter={usd} tick={tick} stroke="#D8DEE5" width={54} />
      <YAxis yAxisId="right" orientation="right" tickFormatter={usd} tick={tick}
        stroke="#D8DEE5" width={54} />
      <Tooltip formatter={(v) => usd(v)} labelFormatter={(t) => `Period ending year ${t}`} />
      <Bar yAxisId="left" dataKey="contribution" name={label(chart, "contribution")}
        fill={COLORS.cva} isAnimationActive={false} />
      <Line yAxisId="right" dataKey="cumulative" name={label(chart, "cumulative")}
        stroke={COLORS.ink} strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
    </Frame>
  );
}

const RENDERERS = {
  exposure_profile: ExposureProfile,
  default_probability: DefaultProbability,
  cva_contribution: CvaContribution,
};

export default function AgentChart({ chart }) {
  const Renderer = RENDERERS[chart.id];
  if (!Renderer) return null; // unknown id: skip rather than crash
  return (
    <figure className="agent-chart">
      <div className="chart-title">{chart.title}</div>
      <Renderer chart={chart} />
      <figcaption className="agent-caption">{chart.caption}</figcaption>
    </figure>
  );
}
