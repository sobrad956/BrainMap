import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
} from "cursor/canvas";

type Trace = {
  mouse: string;
  eid: string;
  trial_index: number;
  region: string;
  acronym: string;
  axial_um: number;
  line_hz: number;
  t: number[];
  raw: number[];
  car: number[];
  paw?: number[];
};
type Psd = {
  mouse: string;
  eid: string;
  region: string;
  acronym: string;
  line_hz: number;
  f: number[];
  raw: number[];
  car: number[];
};
type Spec = {
  event: string;
  mouse: string;
  eid: string;
  region: string;
  line_hz: number;
  t: number[];
  f: number[];
  z: number[][];
};
type Epoch = {
  band: string;
  epoch: string;
  ref: string;
  smooth: string;
  mean: number;
  sem: number;
  n: number;
};
type Spatial = {
  mouse: string;
  eid: string;
  band: string;
  axial: number[];
  power: number[];
  region: string[];
};
type Assoc = {
  band: string;
  ref: string;
  smooth: string;
  r_paw: number | null;
  r_wheel: number | null;
  r_paw_shuffle_mean: number | null;
  r_paw_shuffle_abs95: number | null;
  r_session_swap: number | null;
  n: number;
};
type ExampleTrial = {
  mouse: string;
  lab: string;
  eid: string;
  trial_index: number;
  n_bins: number;
  line_hz: number;
  region: string;
  acronym: string;
  axial_um: number;
  f: number[];
  psd_raw: number[];
  psd_car: number[];
  t: number[];
  band_log: Record<string, number[]>;
  events: {
    stimOn: number;
    goCue: number | null;
    firstMovement: number | null;
    feedback: number | null;
  };
};
type Filters = {
  f: number[];
  hp_01: number[];
  aa_120: number[];
  notch50: number[];
  notch100: number[];
  notch60: number[];
  pre_eu: number[];
  pre_us: number[];
  bands: Record<string, number[]>;
  notes: Record<string, string>;
};
type Inspect = {
  bands: [string, number, number][];
  sessions: {
    eid: string;
    mouse: string;
    lab: string;
    line_hz: number;
    n_trials: number;
    n_channels: number;
  }[];
  traces: Trace[];
  psd: Psd[];
  spectrograms: Spec[];
  epoch_power: Epoch[];
  spatial: Spatial[];
  assoc: Assoc[];
  ref_smooth: {
    ref: string;
    smooth: string;
    high_gamma_paw_r: number | null;
    beta_paw_r: number | null;
    n: number;
  }[];
  example_trials?: ExampleTrial[];
  filters?: Filters;
  corpus?: {
    n_sessions_done: number;
    n_sessions_total: number;
    n_trials_done: number;
    n_trials_total: number;
  };
};

const DATA: Inspect = __DATA__;

const BAND_EDGES: [string, number, number][] = [
  ["δ", 0.5, 4],
  ["θ", 4, 8],
  ["α", 8, 13],
  ["β", 13, 30],
  ["low γ", 30, 55],
  ["high γ", 65, 100],
];

const BAND_LABEL: Record<string, string> = {
  delta: "Delta 0.5–4 Hz",
  theta: "Theta 4–8 Hz",
  alpha: "Alpha 8–13 Hz",
  beta: "Beta 13–30 Hz",
  low_gamma: "Low gamma 30–55 Hz",
  high_gamma: "High gamma 65–100 Hz",
};

const EVENT_TITLE: Record<string, string> = {
  stimOn: "Stimulus onset",
  goCue: "Go cue",
  firstMovement: "First movement",
  feedback: "Feedback",
};

const PLOT = {
  bg: "#ffffff",
  ink: "#000000",
  muted: "#000000",
  grid: "#e4e4e4",
  axis: "#000000",
  shadeA: "#d9e2ec",
  shadeB: "#eee6d9",
  shadeStop: "#ececec",
  unref: "#5c5c5c",
  car: "#0d47a1",
  us: "#b71c1c",
  event: "#555555",
  line: ["#0d47a1", "#b71c1c", "#1b5e20", "#e65100", "#4a148c", "#006064"],
};

let plotUid = 0;
function nextPlotId() {
  plotUid += 1;
  return `lfp-plot-${plotUid}`;
}

function fmt(v: number | null | undefined, d = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(d);
}

function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 1;
    return niceTicks(min - pad, max + pad, count);
  }
  if (min > max) {
    const tmp = min;
    min = max;
    max = tmp;
  }
  const span = max - min;
  const raw = span / Math.max(count - 1, 1);
  const exp = Math.floor(Math.log10(raw));
  const mag = 10 ** exp;
  const err = raw / mag;
  const step = mag * (err < 1.5 ? 1 : err < 3 ? 2 : err < 7 ? 5 : 10);
  const start = Math.ceil((min - 1e-12) / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-9; v += step) {
    ticks.push(Number(v.toPrecision(12)));
    if (ticks.length > 12) break;
  }
  return ticks.length >= 2 ? ticks : [min, max];
}

function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) {
    return Number.isInteger(v) || Math.abs(v - Math.round(v)) < 1e-6
      ? String(Math.round(v))
      : v.toFixed(1);
  }
  if (a >= 1) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(2);
  return v.toExponential(0);
}

function sortXY(x: number[], y: number[]) {
  const pts = x
    .map((xv, i) => [xv, y[i]] as [number, number])
    .filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  pts.sort((a, b) => a[0] - b[0]);
  return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) };
}

function PlotPanel(props: { children?: any }) {
  return (
    <div
      style={{
        minWidth: 0,
        overflow: "hidden",
        background: PLOT.bg,
        padding: 12,
        border: "1px solid #d6d6d6",
        color: "#000000",
      }}
    >
      {props.children}
    </div>
  );
}

function PlotLabel(props: {
  x?: number;
  y?: number;
  fontSize?: number;
  textAnchor?: "start" | "middle" | "end";
  transform?: string;
  children?: any;
}) {
  return (
    <text
      x={props.x}
      y={props.y}
      fontSize={props.fontSize ?? 10}
      textAnchor={props.textAnchor}
      transform={props.transform}
      fill="#000000"
      stroke="none"
      style={{ fill: "#000000 !important", color: "#000000" }}
    >
      {props.children}
    </text>
  );
}

function Legend(props: { items: { name: string; color: string; dash?: string }[] }) {
  return (
    <Row gap={12} wrap>
      {props.items.map((it) => (
        <Row key={it.name} gap={6} align="center">
          <span
            style={{
              width: 18,
              borderTop: it.dash ? `2px dashed ${it.color}` : `2px solid ${it.color}`,
            }}
          />
          <span style={{ color: "#000000", fontSize: 12 }}>{it.name}</span>
        </Row>
      ))}
    </Row>
  );
}

function PolylinePlot(props: {
  series: { x: number[]; y: number[]; color: string; dash?: string; name?: string }[];
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
  height?: number;
  xLabel: string;
  yLabel: string;
  vlines?: { x: number; label: string; color: string }[];
  bands?: { x0: number; x1: number; color: string }[];
}) {
  const clipId = nextPlotId();
  const W = 640;
  const H = props.height ?? 220;
  const l = 64;
  const r = 16;
  const t = 14;
  const b = 54;
  const innerW = W - l - r;
  const innerH = H - b - t;
  const xs = props.series.flatMap((s) => s.x).filter((v) => Number.isFinite(v));
  const ys = props.series.flatMap((s) => s.y).filter((v) => Number.isFinite(v));
  let xMin = props.xMin ?? (xs.length ? Math.min(...xs) : 0);
  let xMax = props.xMax ?? (xs.length ? Math.max(...xs) : 1);
  let yMin = props.yMin ?? (ys.length ? Math.min(...ys) : 0);
  let yMax = props.yMax ?? (ys.length ? Math.max(...ys) : 1);
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  if (props.yMin == null && props.yMax == null) {
    const pad = (yMax - yMin) * 0.06;
    yMin -= pad;
    yMax += pad;
  }
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const sx = (x: number) => l + ((x - xMin) / xSpan) * innerW;
  const sy = (y: number) => t + innerH - ((y - yMin) / ySpan) * innerH;
  const xTicks = niceTicks(xMin, xMax, 6);
  const yTicks = niceTicks(yMin, yMax, 5);
  const path = (x: number[], y: number[]) => {
    const parts: string[] = [];
    let started = false;
    for (let i = 0; i < x.length; i++) {
      if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) {
        started = false;
        continue;
      }
      parts.push(`${started ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(y[i]).toFixed(1)}`);
      started = true;
    }
    return parts.join(" ");
  };
  const vlines = (props.vlines ?? []).filter(
    (v) => Number.isFinite(v.x) && v.x >= xMin && v.x <= xMax
  );
  const named = props.series.filter((s) => s.name);
  return (
    <Stack gap={8}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        style={{ display: "block", background: PLOT.bg, color: "#000000" }}
      >
        <rect x={0} y={0} width={W} height={H} fill={PLOT.bg} />
        <defs>
          <clipPath id={clipId}>
            <rect x={l} y={t} width={innerW} height={innerH} />
          </clipPath>
        </defs>
        {xTicks.map((v) => (
          <line
            key={`xg-${v}`}
            x1={sx(v)}
            y1={t}
            x2={sx(v)}
            y2={t + innerH}
            stroke={PLOT.grid}
          />
        ))}
        {yTicks.map((v) => (
          <line
            key={`yg-${v}`}
            x1={l}
            y1={sy(v)}
            x2={W - r}
            y2={sy(v)}
            stroke={PLOT.grid}
          />
        ))}
        <g clipPath={`url(#${clipId})`}>
          {(props.bands ?? []).map((bn, i) => (
            <rect
              key={i}
              x={sx(bn.x0)}
              y={t}
              width={Math.max(1, sx(bn.x1) - sx(bn.x0))}
              height={innerH}
              fill={bn.color}
              opacity={0.55}
            />
          ))}
          {props.series.map((s, i) => (
            <path
              key={i}
              d={path(s.x, s.y)}
              fill="none"
              stroke={s.color}
              strokeWidth={1.6}
              strokeDasharray={s.dash}
            />
          ))}
          {vlines.map((v, i) => (
            <line
              key={`v-${i}`}
              x1={sx(v.x)}
              y1={t}
              x2={sx(v.x)}
              y2={t + innerH}
              stroke={v.color}
              strokeDasharray="4 3"
              strokeWidth={1.2}
            />
          ))}
        </g>
        <line x1={l} y1={t} x2={l} y2={t + innerH} stroke={PLOT.axis} strokeWidth={1.2} />
        <line
          x1={l}
          y1={t + innerH}
          x2={W - r}
          y2={t + innerH}
          stroke={PLOT.axis}
          strokeWidth={1.2}
        />
        {xTicks.map((v) => (
          <g key={`xt-${v}`}>
            <line
              x1={sx(v)}
              y1={t + innerH}
              x2={sx(v)}
              y2={t + innerH + 5}
              stroke={PLOT.axis}
            />
            <PlotLabel x={sx(v)} y={t + innerH + 18} fontSize={10} textAnchor="middle">
              {fmtTick(v)}
            </PlotLabel>
          </g>
        ))}
        {yTicks.map((v) => (
          <g key={`yt-${v}`}>
            <line x1={l - 5} y1={sy(v)} x2={l} y2={sy(v)} stroke={PLOT.axis} />
            <PlotLabel x={l - 8} y={sy(v) + 3} fontSize={10} textAnchor="end">
              {fmtTick(v)}
            </PlotLabel>
          </g>
        ))}
        <PlotLabel x={l + innerW / 2} y={H - 8} fontSize={11} textAnchor="middle">
          {props.xLabel}
        </PlotLabel>
        <PlotLabel
          fontSize={11}
          textAnchor="middle"
          transform={`translate(14 ${t + innerH / 2}) rotate(-90)`}
        >
          {props.yLabel}
        </PlotLabel>
      </svg>
      {vlines.length ? (
        <Row gap={12} wrap>
          {vlines.map((v, i) => (
            <span key={`${v.label}-${i}`} style={{ color: "#000000", fontSize: 12 }}>
              {v.label} @ {v.x.toFixed(2)} s
            </span>
          ))}
        </Row>
      ) : null}
      {named.length ? (
        <Legend
          items={named.map((s) => ({
            name: s.name as string,
            color: s.color,
            dash: s.dash,
          }))}
        />
      ) : null}
    </Stack>
  );
}

function heatFill(v: number, lo: number, hi: number) {
  if (!Number.isFinite(v)) return PLOT.bg;
  if (v >= 0) {
    const u = Math.max(0, Math.min(1, v / (hi || 1)));
    const r = Math.round(255 + (139 - 255) * u);
    const g = Math.round(255 + (26 - 255) * u);
    const b = Math.round(255 + (26 - 255) * u);
    return `rgb(${r},${g},${b})`;
  }
  const u = Math.max(0, Math.min(1, v / (lo || -1)));
  const r = Math.round(255 + (13 - 255) * u);
  const g = Math.round(255 + (71 - 255) * u);
  const b = Math.round(255 + (161 - 255) * u);
  return `rgb(${r},${g},${b})`;
}

function Heatmap(props: { spec: Spec }) {
  const { t, f, z } = props.spec;
  const clipId = nextPlotId();
  const W = 560;
  const H = 240;
  const l = 64;
  const b = 54;
  const top = 14;
  const r = 52;
  const innerW = W - l - r;
  const innerH = H - b - top;
  const nR = z.length;
  const nC = z[0]?.length ?? 0;
  const cw = innerW / Math.max(nC, 1);
  const ch = innerH / Math.max(nR, 1);
  const lo = -1.2;
  const hi = 1.2;
  const tMin = t[0] ?? -0.4;
  const tMax = t[t.length - 1] ?? 0.8;
  const fMin = f[0] ?? 0;
  const fMax = f[f.length - 1] ?? 100;
  const sx = (x: number) => l + ((x - tMin) / (tMax - tMin || 1)) * innerW;
  const sy = (y: number) => top + innerH - ((y - fMin) / (fMax - fMin || 1)) * innerH;
  const xTicks = niceTicks(tMin, tMax, 5);
  const yTicks = niceTicks(fMin, fMax, 5);
  const barX = W - 36;
  const barY = top;
  const barH = innerH;
  const barStops = niceTicks(lo, hi, 5);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      role="img"
      style={{ display: "block", background: PLOT.bg, color: "#000000" }}
    >
      <rect x={0} y={0} width={W} height={H} fill={PLOT.bg} />
      <defs>
        <clipPath id={clipId}>
          <rect x={l} y={top} width={innerW} height={innerH} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        {z.map((row, ri) =>
          row.map((v, ci) => (
            <rect
              key={`${ri}-${ci}`}
              x={l + ci * cw}
              y={top + (nR - 1 - ri) * ch}
              width={Math.ceil(cw + 0.5)}
              height={Math.ceil(ch + 0.5)}
              fill={heatFill(v, lo, hi)}
            />
          ))
        )}
        <line
          x1={sx(0)}
          y1={top}
          x2={sx(0)}
          y2={top + innerH}
          stroke={PLOT.axis}
          strokeDasharray="3 3"
        />
      </g>
      <line x1={l} y1={top} x2={l} y2={top + innerH} stroke={PLOT.axis} strokeWidth={1.2} />
      <line
        x1={l}
        y1={top + innerH}
        x2={l + innerW}
        y2={top + innerH}
        stroke={PLOT.axis}
        strokeWidth={1.2}
      />
      {xTicks.map((v) => (
        <g key={`xt-${v}`}>
          <line
            x1={sx(v)}
            y1={top + innerH}
            x2={sx(v)}
            y2={top + innerH + 5}
            stroke={PLOT.axis}
          />
          <PlotLabel x={sx(v)} y={top + innerH + 18} fontSize={10} textAnchor="middle">
            {fmtTick(v)}
          </PlotLabel>
        </g>
      ))}
      {yTicks.map((v) => (
        <g key={`yt-${v}`}>
          <line x1={l - 5} y1={sy(v)} x2={l} y2={sy(v)} stroke={PLOT.axis} />
          <PlotLabel x={l - 8} y={sy(v) + 3} fontSize={10} textAnchor="end">
            {fmtTick(v)}
          </PlotLabel>
        </g>
      ))}
      <PlotLabel x={l + innerW / 2} y={H - 8} fontSize={11} textAnchor="middle">
        Time from event (s)
      </PlotLabel>
      <PlotLabel
        fontSize={11}
        textAnchor="middle"
        transform={`translate(14 ${top + innerH / 2}) rotate(-90)`}
      >
        Frequency (Hz)
      </PlotLabel>
      {Array.from({ length: 24 }, (_, i) => {
        const u = 1 - i / 23;
        const val = lo + u * (hi - lo);
        return (
          <rect
            key={i}
            x={barX}
            y={barY + (i / 24) * barH}
            width={10}
            height={barH / 24 + 0.5}
            fill={heatFill(val, lo, hi)}
          />
        );
      })}
      <rect x={barX} y={barY} width={10} height={barH} fill="none" stroke={PLOT.axis} />
      {barStops.map((v) => (
        <PlotLabel
          key={v}
          x={barX + 14}
          y={barY + ((hi - v) / (hi - lo)) * barH + 3}
          fontSize={9}
        >
          {fmtTick(v)}
        </PlotLabel>
      ))}
    </svg>
  );
}

function WhiteBarChart(props: {
  categories: string[];
  series: { name: string; data: number[]; color: string }[];
  xLabel: string;
  yLabel: string;
  height?: number;
}) {
  const clipId = nextPlotId();
  const W = 680;
  const H = props.height ?? 260;
  const l = 64;
  const r = 16;
  const t = 14;
  const b = 58;
  const innerW = W - l - r;
  const innerH = H - b - t;
  const vals = props.series.flatMap((s) => s.data).filter((v) => Number.isFinite(v));
  let yMin = vals.length ? Math.min(0, ...vals) : -1;
  let yMax = vals.length ? Math.max(0, ...vals) : 1;
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.08;
  yMin -= pad;
  yMax += pad;
  const yTicks = niceTicks(yMin, yMax, 5);
  const nCat = Math.max(props.categories.length, 1);
  const nSer = Math.max(props.series.length, 1);
  const groupW = innerW / nCat;
  const barW = Math.min(16, (groupW * 0.72) / nSer);
  const groupPad = (groupW - barW * nSer) / 2;
  const sy = (y: number) => t + innerH - ((y - yMin) / (yMax - yMin || 1)) * innerH;
  const y0 = sy(0);
  return (
    <Stack gap={8}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        style={{ display: "block", background: PLOT.bg, color: "#000000" }}
      >
        <rect x={0} y={0} width={W} height={H} fill={PLOT.bg} />
        <defs>
          <clipPath id={clipId}>
            <rect x={l} y={t} width={innerW} height={innerH} />
          </clipPath>
        </defs>
        {yTicks.map((v) => (
          <line
            key={`yg-${v}`}
            x1={l}
            y1={sy(v)}
            x2={W - r}
            y2={sy(v)}
            stroke={PLOT.grid}
          />
        ))}
        <g clipPath={`url(#${clipId})`}>
          <line x1={l} y1={y0} x2={W - r} y2={y0} stroke={PLOT.axis} />
          {props.series.map((s, si) =>
            s.data.map((v, ci) => {
              const x = l + ci * groupW + groupPad + si * barW;
              const yTop = Math.min(sy(v), y0);
              const h = Math.max(1, Math.abs(sy(v) - y0));
              return (
                <rect
                  key={`${si}-${ci}`}
                  x={x}
                  y={yTop}
                  width={Math.max(1, barW - 2)}
                  height={h}
                  fill={s.color}
                />
              );
            })
          )}
        </g>
        <line x1={l} y1={t} x2={l} y2={t + innerH} stroke={PLOT.axis} strokeWidth={1.2} />
        <line
          x1={l}
          y1={t + innerH}
          x2={W - r}
          y2={t + innerH}
          stroke={PLOT.axis}
          strokeWidth={1.2}
        />
        {yTicks.map((v) => (
          <g key={`yt-${v}`}>
            <line x1={l - 5} y1={sy(v)} x2={l} y2={sy(v)} stroke={PLOT.axis} />
            <PlotLabel x={l - 8} y={sy(v) + 3} fontSize={10} textAnchor="end">
              {fmtTick(v)}
            </PlotLabel>
          </g>
        ))}
        {props.categories.map((cat, i) => (
          <PlotLabel
            key={cat}
            x={l + i * groupW + groupW / 2}
            y={t + innerH + 18}
            fontSize={10}
            textAnchor="middle"
          >
            {cat}
          </PlotLabel>
        ))}
        <PlotLabel x={l + innerW / 2} y={H - 8} fontSize={11} textAnchor="middle">
          {props.xLabel}
        </PlotLabel>
        <PlotLabel
          fontSize={11}
          textAnchor="middle"
          transform={`translate(14 ${t + innerH / 2}) rotate(-90)`}
        >
          {props.yLabel}
        </PlotLabel>
      </svg>
      <Legend items={props.series.map((s) => ({ name: s.name, color: s.color }))} />
    </Stack>
  );
}

function trialVlines(ex: ExampleTrial) {
  return (
    [
      ["stim", ex.events.stimOn],
      ["go", ex.events.goCue],
      ["move", ex.events.firstMovement],
      ["fb", ex.events.feedback],
    ] as [string, number | null][]
  )
    .filter(([label, x]) => {
      if (x == null || !Number.isFinite(x)) return false;
      if (label === "go" && Math.abs(x) < 0.05) return false;
      return true;
    })
    .map(([label, x]) => ({
      x: x as number,
      label,
      color: PLOT.event,
    }));
}

export default function LfpRightContraQc() {
  const mice = Array.from(new Set(DATA.sessions.map((s) => s.mouse)));
  const [mouse, setMouse] = useCanvasState("lfpqc.mouse", mice[0] ?? "");
  const regions = Array.from(
    new Set(DATA.traces.filter((t) => t.mouse === mouse).map((t) => t.region))
  );
  const [region, setRegion] = useCanvasState("lfpqc.region", regions[0] ?? "MOp");
  const [ref, setRef] = useCanvasState("lfpqc.ref", "car");
  const [smooth, setSmooth] = useCanvasState("lfpqc.smooth", "s20");

  const trace =
    DATA.traces.find((t) => t.mouse === mouse && t.region === region) ??
    DATA.traces.find((t) => t.mouse === mouse) ??
    DATA.traces[0];
  const psd =
    DATA.psd.find((t) => t.mouse === mouse && t.region === region) ??
    DATA.psd.find((t) => t.mouse === mouse) ??
    DATA.psd[0];
  const specs = DATA.spectrograms.filter((s) => s.mouse === mouse);
  const epochs = DATA.epoch_power.filter((e) => e.ref === ref && e.smooth === smooth);
  const spatial = DATA.spatial.filter((s) => s.mouse === mouse);
  const assoc = DATA.assoc.filter((a) => a.ref === ref && a.smooth === smooth);
  const nTrials = DATA.sessions.reduce((a, s) => a + s.n_trials, 0);
  const n50 = DATA.sessions.filter((s) => s.line_hz === 50).length;
  const n60 = DATA.sessions.filter((s) => s.line_hz === 60).length;
  const filters = DATA.filters;

  const epochCats = ["baseline", "preparation", "movement"];
  const epochSeries = DATA.bands.map(([name], i) => ({
    name: BAND_LABEL[name] ?? name,
    color: PLOT.line[i % PLOT.line.length],
    data: epochCats.map((ep) => {
      const row = epochs.find((e) => e.band === name && e.epoch === ep);
      return row ? row.mean : 0;
    }),
  }));

  const assocCats = DATA.bands.map(([n]) => n);
  const assocSeries = [
    {
      name: "r paw",
      color: PLOT.car,
      data: assocCats.map((b) => assoc.find((a) => a.band === b)?.r_paw ?? 0),
    },
    {
      name: "r wheel",
      color: PLOT.us,
      data: assocCats.map((b) => assoc.find((a) => a.band === b)?.r_wheel ?? 0),
    },
    {
      name: "shuffle paw",
      color: PLOT.unref,
      data: assocCats.map((b) => assoc.find((a) => a.band === b)?.r_paw_shuffle_mean ?? 0),
    },
    {
      name: "session swap",
      color: PLOT.line[2],
      data: assocCats.map((b) => assoc.find((a) => a.band === b)?.r_session_swap ?? 0),
    },
  ];

  return (
    <Stack gap={28} style={{ padding: 20 }}>
      <Stack gap={6}>
        <H1>Motor-cortex LFP features, ModelDataRightContra</H1>
        <Text tone="secondary">
          Canonical band power on the same 20 ms bins as spikesorted units.
          Voltage is SpikeGLX LFP streamed from OpenAlyx S3, not the 30 kHz
          action-potential band.
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat label="QC sessions" value={String(DATA.sessions.length)} tone="info" />
        <Stat label="QC trials" value={String(nTrials)} />
        <Stat label="Line notch" value={`${n50}×50 Hz, ${n60}×60 Hz`} />
        <Stat
          label="Corpus ingest"
          value={`${DATA.corpus?.n_sessions_done ?? DATA.sessions.length}/${DATA.corpus?.n_sessions_total ?? 52}`}
        />
      </Grid>

      <Callout tone="info" title="What an LFP is">
        A local field potential is the slow electrical voltage picked up by a
        probe contact. It is the summed activity of many neurons near that
        contact, not a single spike. We split that voltage into frequency
        bands the way a chord splits into notes, then measure how strong each
        note is over the trial.
      </Callout>

      <Stack gap={8}>
        <H2>Pipeline, from raw voltage to features</H2>
        <Text tone="secondary">
          Each step below is applied in order. Later plots are the QC for that
          step.
        </Text>
        <Table
          headers={["Step", "What happens", "Where to look"]}
          columnAlign={["left", "left", "left"]}
          rows={[
            [
              "1. Stream LFP",
              "Read the ~2500 Hz voltage around each trial from the Neuropixels LFP stream (not the 30 kHz spike stream).",
              "Raw vs CAR voltage",
            ],
            [
              "2. Anti-alias + resample",
              "Low-pass at 120 Hz, then drop to 250 samples/s. That is fast enough for 100 Hz gamma and slow enough to store.",
              "Filters: 120 Hz anti-alias",
            ],
            [
              "3. High-pass 0.1 Hz",
              "Remove slow drift and electrode offset so the trace sits around 0 µV.",
              "Filters: 0.1 Hz high-pass",
            ],
            [
              "4. Notch line noise",
              "EU labs: cut 50 Hz and 100 Hz. US labs: cut 60 Hz. Building electricity leaks into the recording.",
              "Filters + EU/US PSDs",
            ],
            [
              "5. Reference (CAR / bipolar)",
              "Subtract the median of all channels (CAR) or a neighbor 5–45 µm away (bipolar). Shared noise drops; local brain signal remains.",
              "Raw vs CAR; PSD before/after",
            ],
            [
              "6. Bandpass",
              "Keep only one frequency range at a time (delta … high gamma). The 55–65 Hz gap is unused so 60 Hz line noise cannot leak into gamma.",
              "Canonical bandpass filters",
            ],
            [
              "7. Hilbert power",
              "Turn the oscillation into an always-positive envelope, then square it. That is instantaneous band power (µV²).",
              "Time-resolved band power",
            ],
            [
              "8. 20 ms bins",
              "Average power in the same 20 ms bins as spikes and paw speed, indexed by eid + trial_index + n_bins.",
              "Epoch bars; paw association",
            ],
          ]}
          striped
        />
      </Stack>

      <Row gap={12} wrap>
        {DATA.sessions.map((s) => (
          <Pill
            key={s.eid}
            active={s.mouse === mouse}
            tone={s.line_hz === 60 ? "warning" : "info"}
          >
            {s.mouse} · {s.lab} · {s.line_hz} Hz · {s.n_trials} trials
          </Pill>
        ))}
      </Row>

      <Row gap={16} wrap align="center">
        <Text size="small" tone="secondary">
          Mouse
        </Text>
        <Select
          value={mouse}
          onChange={setMouse}
          options={mice.map((m) => ({ value: m, label: m }))}
        />
        <Text size="small" tone="secondary">
          Channel
        </Text>
        <Select
          value={region}
          onChange={setRegion}
          options={["MOp", "MOs", "other"]
            .filter((r) => DATA.traces.some((t) => t.region === r))
            .map((r) => ({ value: r, label: r }))}
        />
        <Text size="small" tone="secondary">
          Reference
        </Text>
        <Select
          value={ref}
          onChange={setRef}
          options={[
            { value: "car", label: "Median CAR" },
            { value: "bip", label: "Adjacent bipolar" },
          ]}
        />
        <Text size="small" tone="secondary">
          Smooth
        </Text>
        <Select
          value={smooth}
          onChange={setSmooth}
          options={[
            { value: "s20", label: "20 ms bin mean" },
            { value: "s100", label: "σ = 100 ms" },
          ]}
        />
      </Row>

      {filters ? (
        <Stack gap={20}>
          <Divider />
          <H2>Filters and stop bands</H2>
          <Text tone="secondary">
            These curves are the actual filters applied to the voltage. Gain in
            dB: 0 dB means “let this frequency through unchanged,” −3 dB is
            half power, −80 dB is essentially blocked. Shaded regions are
            stopbands or unused frequencies. Zero-phase filtfilt, so the
            curves are |H|².
          </Text>

          <Stack gap={8}>
            <H3>Preprocessing at 250 Hz</H3>
            <Text size="small" tone="tertiary">
              High-pass, anti-alias, and mains notches on one axis so you can
              see which frequencies each stage kills. Source: SOS stages in
              build_lfp_features.py.
            </Text>
            <PlotPanel>
              <PolylinePlot
                height={240}
                xMin={0}
                xMax={125}
                yMin={-80}
                yMax={5}
                xLabel="Frequency (Hz)"
                yLabel="Gain (dB)"
                bands={[
                  { x0: 0, x1: 0.1, color: PLOT.shadeStop },
                  { x0: 120, x1: 125, color: PLOT.shadeStop },
                ]}
                series={[
                  {
                    name: "0.1 Hz high-pass",
                    x: filters.f,
                    y: filters.hp_01,
                    color: PLOT.unref,
                  },
                  {
                    name: "120 Hz anti-alias",
                    x: filters.f,
                    y: filters.aa_120,
                    color: PLOT.event,
                    dash: "4 3",
                  },
                  {
                    name: "50 Hz notch",
                    x: filters.f,
                    y: filters.notch50,
                    color: PLOT.car,
                  },
                  {
                    name: "100 Hz harmonic",
                    x: filters.f,
                    y: filters.notch100,
                    color: PLOT.car,
                    dash: "4 3",
                  },
                  {
                    name: "60 Hz notch",
                    x: filters.f,
                    y: filters.notch60,
                    color: PLOT.us,
                  },
                ]}
                vlines={[
                  { x: 50, label: "50 Hz mains", color: PLOT.event },
                  { x: 60, label: "60 Hz mains", color: PLOT.event },
                ]}
              />
            </PlotPanel>
          </Stack>

          <Stack gap={8}>
            <H3>Canonical bandpass filters (one panel per band)</H3>
            <Text size="small" tone="tertiary">
              Each panel is one 4th-order Butterworth bandpass. The shaded
              strip is that band’s pass region. High gamma starts at 65 Hz so
              it sits above the unused 55–65 Hz gap around US mains.
            </Text>
            <Grid columns={2} gap={20}>
              {DATA.bands.map(([name, lo, hi]) => (
                <Stack key={name} gap={6} style={{ minWidth: 0 }}>
                  <H3>{BAND_LABEL[name] ?? name}</H3>
                  <PlotPanel>
                    <PolylinePlot
                      height={200}
                      xMin={0}
                      xMax={125}
                      yMin={-80}
                      yMax={5}
                      xLabel="Frequency (Hz)"
                      yLabel="Gain (dB)"
                      bands={[{ x0: lo, x1: hi, color: PLOT.shadeA }]}
                      series={[
                        {
                          x: filters.f,
                          y: filters.bands[name],
                          color: PLOT.car,
                        },
                      ]}
                    />
                  </PlotPanel>
                </Stack>
              ))}
            </Grid>
          </Stack>

          <Stack gap={8}>
            <H3>Combined high-pass + lab notch</H3>
            <Text size="small" tone="tertiary">
              What actually runs before bandpass: 0.1 Hz high-pass plus the
              lab’s mains notch. EU also notches 100 Hz. US 120 Hz is skipped
              because it sits above 0.45 × 250 Hz. {filters.notes.notch}
            </Text>
            <PlotPanel>
              <PolylinePlot
                height={240}
                xMin={0}
                xMax={125}
                yMin={-80}
                yMax={5}
                xLabel="Frequency (Hz)"
                yLabel="Gain (dB)"
                bands={[
                  { x0: 0, x1: 0.1, color: PLOT.shadeStop },
                  { x0: 55, x1: 65, color: PLOT.shadeB },
                  { x0: 120, x1: 125, color: PLOT.shadeStop },
                ]}
                series={[
                  {
                    name: "EU: HP + 50 & 100 Hz",
                    x: filters.f,
                    y: filters.pre_eu,
                    color: PLOT.car,
                  },
                  {
                    name: "US: HP + 60 Hz",
                    x: filters.f,
                    y: filters.pre_us,
                    color: PLOT.us,
                    dash: "4 3",
                  },
                ]}
                vlines={[
                  { x: 50, label: "50 Hz", color: PLOT.event },
                  { x: 60, label: "60 Hz", color: PLOT.event },
                ]}
              />
            </PlotPanel>
          </Stack>
        </Stack>
      ) : null}

      {DATA.example_trials && DATA.example_trials.length ? (
        <Stack gap={16}>
          <Divider />
          <H2>Example-trial power spectra, EU vs US</H2>
          <Text tone="secondary">
            A power spectrum asks: how much energy lives at each frequency?
            X is frequency in Hz, Y is log10 power (higher = stronger). A
            spike at 50 or 60 Hz is wall-power contamination, not brain. CAR
            (solid) should shrink that spike relative to the unreferenced
            trace (dashed). Shaded strips are the canonical bands.
          </Text>
          <Grid columns={2} gap={24}>
            {DATA.example_trials.map((ex) => (
              <Stack key={`${ex.eid}-${ex.trial_index}`} gap={8} style={{ minWidth: 0 }}>
                <H3>
                  {ex.mouse} · {ex.lab} · {ex.line_hz} Hz · trial {ex.trial_index}
                </H3>
                <Text size="small" tone="tertiary">
                  {ex.region} {ex.acronym} · {ex.n_bins} bins · eid {ex.eid}
                </Text>
                <PlotPanel>
                  <PolylinePlot
                    height={220}
                    xMin={0}
                    xMax={120}
                    xLabel="Frequency (Hz)"
                    yLabel="log10 power (a.u.)"
                    bands={BAND_EDGES.map(([, lo, hi], i) => ({
                      x0: lo,
                      x1: hi,
                      color: i % 2 === 0 ? PLOT.shadeA : PLOT.shadeB,
                    }))}
                    series={[
                      {
                        name: "Unreferenced",
                        x: ex.f,
                        y: ex.psd_raw,
                        color: PLOT.unref,
                        dash: "4 3",
                      },
                      {
                        name: "Median CAR",
                        x: ex.f,
                        y: ex.psd_car,
                        color: PLOT.car,
                      },
                    ]}
                    vlines={[
                      {
                        x: ex.line_hz,
                        label: `${ex.line_hz} Hz mains`,
                        color: PLOT.event,
                      },
                    ]}
                  />
                </PlotPanel>
              </Stack>
            ))}
          </Grid>
        </Stack>
      ) : null}

      {DATA.example_trials && DATA.example_trials.length ? (
        <Stack gap={16}>
          <Divider />
          <H2>Time-resolved band power, trial-aligned</H2>
          <Text tone="secondary">
            Each panel is one frequency band on one trial. X is time from
            stimulus onset (seconds). Y is log10 Hilbert power after median
            CAR, averaged into 20 ms bins. A rise after movement in high
            gamma is a typical motor-cortex signature; a bump locked to the
            50/60 Hz line would be leftover artifact. Dashed verticals are
            task events; their times are listed under each plot.
          </Text>
          {DATA.example_trials.map((ex) => {
            const vlines = trialVlines(ex);
            return (
              <Card key={`band-${ex.eid}-${ex.trial_index}`}>
                <CardHeader
                  trailing={`${ex.lab} · ${ex.line_hz} Hz · ${ex.n_bins} bins`}
                >
                  {ex.mouse} · trial {ex.trial_index} · {ex.region} {ex.acronym}
                </CardHeader>
                <CardBody>
                  <Grid columns={2} gap={24}>
                    {DATA.bands.map(([name, lo, hi]) => (
                      <Stack key={name} gap={6} style={{ minWidth: 0 }}>
                        <H3>{BAND_LABEL[name] ?? `${name} ${lo}–${hi} Hz`}</H3>
                        <PlotPanel>
                          <PolylinePlot
                            height={190}
                            xLabel="Time from stimOn (s)"
                            yLabel="log10 power (µV²)"
                            series={[
                              {
                                x: ex.t,
                                y: ex.band_log[name] ?? [],
                                color: PLOT.car,
                              },
                            ]}
                            vlines={vlines}
                          />
                        </PlotPanel>
                      </Stack>
                    ))}
                  </Grid>
                  <Text size="small" tone="tertiary">
                    Source: ModelDataRightContraLFP power_car_s20, mean over
                    motor channels, same 20 ms grid as the spike corpus.
                  </Text>
                </CardBody>
              </Card>
            );
          })}
        </Stack>
      ) : null}

      <Divider />

      <H2>1. Raw LFP versus CAR</H2>
      <Text tone="secondary">
        This is the voltage itself, in microvolts, on a {trace?.region}{" "}
        channel ({trace?.acronym}) from {trace?.mouse}, trial {trace?.trial_index}.
        Time 0 is stimulus onset. The dashed trace still contains noise
        shared across the probe (mains, motion, cable). The solid trace had
        that shared signal subtracted (median common-average reference). Paw
        speed on the same trial is the behavioral target at 20 ms.
      </Text>
      {trace ? (
        <Grid columns={2} gap={24}>
          <Stack gap={8} style={{ minWidth: 0 }}>
            <H3>Voltage (µV)</H3>
            <PlotPanel>
              <PolylinePlot
                height={220}
                series={[
                  {
                    name: "Unreferenced",
                    x: trace.t,
                    y: trace.raw,
                    color: PLOT.unref,
                    dash: "4 3",
                  },
                  {
                    name: "Median CAR",
                    x: trace.t,
                    y: trace.car,
                    color: PLOT.car,
                  },
                ]}
                xLabel="Time from stimOn (s)"
                yLabel="Voltage (µV)"
                vlines={[{ x: 0, label: "stim", color: PLOT.event }]}
              />
            </PlotPanel>
          </Stack>
          <Stack gap={8} style={{ minWidth: 0 }}>
            <H3>Paw speed (px/s)</H3>
            {trace.paw && trace.paw.length ? (
              <PlotPanel>
                <PolylinePlot
                  height={220}
                  series={[
                    {
                      name: "Paw speed",
                      x: trace.paw.map((_, i) => i * 0.02),
                      y: trace.paw,
                      color: PLOT.us,
                    },
                  ]}
                  xLabel="Time from stimOn (s)"
                  yLabel="Paw speed (px/s)"
                  vlines={[{ x: 0, label: "stim", color: PLOT.event }]}
                />
              </PlotPanel>
            ) : (
              <Text tone="secondary">No paw trace on this trial.</Text>
            )}
            <Text size="small" tone="tertiary">
              Source: ModelDataRightContra lp_speed, 20 ms bins. Compare
              motion here with voltage above — shared wiggles can be artifact.
            </Text>
          </Stack>
        </Grid>
      ) : null}

      <Divider />

      <H2>2. PSD before and after referencing</H2>
      <Text tone="secondary">
        Welch log10 power spectral density of the 250 Hz traces for the
        selected mouse/channel. If CAR is working, the dashed peak at this
        session’s mains frequency ({psd?.line_hz} Hz) should shrink on the
        solid curve. Shaded bands are the frequencies we later keep.
      </Text>
      {psd ? (
        <PlotPanel>
          <PolylinePlot
            height={240}
            series={[
              {
                name: "Unreferenced",
                x: psd.f,
                y: psd.raw,
                color: PLOT.unref,
                dash: "4 3",
              },
              {
                name: "Median CAR",
                x: psd.f,
                y: psd.car,
                color: PLOT.car,
              },
            ]}
            xMin={0}
            xMax={120}
            xLabel="Frequency (Hz)"
            yLabel="log10 PSD (a.u.)"
            bands={BAND_EDGES.map(([, lo, hi], i) => ({
              x0: lo,
              x1: hi,
              color: i % 2 === 0 ? PLOT.shadeA : PLOT.shadeB,
            }))}
            vlines={[
              {
                x: psd.line_hz,
                label: `${psd.line_hz} Hz mains`,
                color: PLOT.event,
              },
            ]}
          />
        </PlotPanel>
      ) : null}
      <Row gap={8} wrap>
        {BAND_EDGES.map(([name, lo, hi]) => (
          <Pill key={name}>
            {name} {lo}–{hi} Hz
          </Pill>
        ))}
      </Row>

      <Divider />

      <H2>3. Spectrograms aligned to task events</H2>
      <Text tone="secondary">
        A spectrogram is a power spectrum that also moves through time. Color
        is change in log-power versus the pre-event baseline (white ≈ no
        change, red = more power than baseline, blue = less). The dashed
        vertical is the aligned event. Mouse {mouse}.
      </Text>
      <Grid columns={2} gap={24}>
        {["stimOn", "goCue", "firstMovement", "feedback"].map((ev) => {
          const sp =
            specs.find((s) => s.event === ev) ??
            DATA.spectrograms.find((s) => s.event === ev);
          if (!sp) return null;
          return (
            <Stack key={ev} gap={8} style={{ minWidth: 0 }}>
              <H3>
                {EVENT_TITLE[ev] ?? ev} · {sp.mouse} · {sp.region}
              </H3>
              <PlotPanel>
                <Heatmap spec={sp} />
              </PlotPanel>
              <Text size="small" tone="tertiary">
                Color scale: Δ log-power vs the first fifth of the window.
                Dashed line is time 0 for {EVENT_TITLE[ev] ?? ev}.
              </Text>
            </Stack>
          );
        })}
      </Grid>

      <Divider />

      <H2>4. Epoch log-power by band</H2>
      <Text tone="secondary">
        Mean motor-channel log10 power (µV²) in three task epochs: baseline
        (before stim), preparation (stim to go / first move), and movement.
        Current view: {ref === "car" ? "median CAR" : "bipolar"},{" "}
        {smooth === "s20" ? "20 ms bins" : "σ = 100 ms"}. If movement
        genuinely engages cortex, movement bars should differ from baseline
        — especially in beta and high gamma.
      </Text>
      <PlotPanel>
        <WhiteBarChart
          categories={epochCats}
          series={epochSeries}
          height={280}
          xLabel="Trial epoch"
          yLabel="Mean log10 power (µV²)"
        />
      </PlotPanel>
      <Table
        headers={["Band", "Baseline", "Preparation", "Movement", "n"]}
        columnAlign={["left", "right", "right", "right", "right"]}
        rows={DATA.bands.map(([name]) => {
          const b = epochs.find((e) => e.band === name && e.epoch === "baseline");
          const p = epochs.find((e) => e.band === name && e.epoch === "preparation");
          const m = epochs.find((e) => e.band === name && e.epoch === "movement");
          return [
            BAND_LABEL[name] ?? name,
            fmt(b?.mean, 2),
            fmt(p?.mean, 2),
            fmt(m?.mean, 2),
            String(b?.n ?? m?.n ?? "—"),
          ];
        })}
        striped
      />

      <Divider />

      <H2>5. Band power along the probe</H2>
      <Text tone="secondary">
        Mean movement-epoch CAR log-power versus channel depth for {mouse},
        one panel per canonical band. X is axial depth along the shank (µm).
        Y is log10 power. Motor cortex (MOp, MOs) should not look identical
        to every other region on the same probe — that would suggest a
        global artifact, not a local signal.
      </Text>
      <Grid columns={2} gap={24}>
        {DATA.bands.map(([band]) => {
          const row = spatial.find((s) => s.band === band);
          if (!row) return null;
          const by = (reg: string) => {
            const xs: number[] = [];
            const ys: number[] = [];
            row.region.forEach((r, i) => {
              if (r === reg) {
                xs.push(row.axial[i]);
                ys.push(row.power[i]);
              }
            });
            return sortXY(xs, ys);
          };
          const mop = by("MOp");
          const mos = by("MOs");
          const oth = by("other");
          return (
            <Stack key={band} gap={8} style={{ minWidth: 0 }}>
              <H3>{BAND_LABEL[band] ?? band}</H3>
              <PlotPanel>
                <PolylinePlot
                  height={220}
                  series={[
                    {
                      name: "Other regions",
                      x: oth.x,
                      y: oth.y,
                      color: PLOT.unref,
                    },
                    {
                      name: "MOp",
                      x: mop.x,
                      y: mop.y,
                      color: PLOT.car,
                    },
                    {
                      name: "MOs",
                      x: mos.x,
                      y: mos.y,
                      color: PLOT.us,
                    },
                  ].filter((s) => s.x.length > 2)}
                  xLabel="Axial depth (µm)"
                  yLabel="log10 power (µV²)"
                />
              </PlotPanel>
            </Stack>
          );
        })}
      </Grid>

      <Divider />

      <H2>6. Band power vs wheel and paw</H2>
      <Text tone="secondary">
        Pearson r across trials between trial-mean motor log-power and mean
        |paw speed| / |wheel velocity|. Shuffle: permute paw labels 80
        times. Session swap: give each trial the band-power of another
        session. A real neural effect should beat shuffle and not collapse
        to session identity. Current view: {ref === "car" ? "CAR" : "bipolar"},{" "}
        {smooth === "s20" ? "20 ms" : "100 ms"}.
      </Text>
      <PlotPanel>
        <WhiteBarChart
          categories={assocCats}
          series={assocSeries}
          height={280}
          xLabel="Frequency band"
          yLabel="Pearson r"
        />
      </PlotPanel>
      <Table
        headers={[
          "Band",
          "r paw",
          "r wheel",
          "shuffle mean",
          "|shuffle| 95%",
          "session swap",
          "n",
        ]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right"]}
        striped
        rows={assoc.map((a) => [
          BAND_LABEL[a.band] ?? a.band,
          fmt(a.r_paw, 3),
          fmt(a.r_wheel, 3),
          fmt(a.r_paw_shuffle_mean, 3),
          fmt(a.r_paw_shuffle_abs95, 3),
          fmt(a.r_session_swap, 3),
          String(a.n),
        ])}
      />

      <H3>Reference × smoothing, high-gamma and beta vs paw</H3>
      <Table
        headers={["Reference", "Smoothing", "high γ · paw r", "β · paw r", "n"]}
        columnAlign={["left", "left", "right", "right", "right"]}
        rows={DATA.ref_smooth.map((r) => [
          r.ref === "car" ? "median CAR" : "bipolar",
          r.smooth === "s20" ? "20 ms bin" : "σ = 100 ms",
          fmt(r.high_gamma_paw_r, 3),
          fmt(r.beta_paw_r, 3),
          String(r.n),
        ])}
        striped
      />
      <Text size="small" tone="tertiary">
        Source: ModelDataRightContraLFP session pickles plus
        ModelDataRightContra paw/wheel. Features are indexed by eid +
        trial_index + n_bins so they can be passed in with spikesorted units.
        Filter curves are the SOS stages used in build_lfp_features.py,
        evaluated as zero-phase gain.
      </Text>
    </Stack>
  );
}
