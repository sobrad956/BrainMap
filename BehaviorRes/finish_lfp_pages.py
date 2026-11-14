#!/usr/bin/env python3
"""Finish the LFP viewer pages: split traces data, fix metrics JS crash, polish UI."""
from pathlib import Path

ROOT = Path("/Users/Stefan/Desktop/BrainMap/BehaviorRes")
TRACES = ROOT / "experiments_clean_traces_lfp.html"
DATA_JS = ROOT / "experiments_clean_traces_lfp_data.js"
METRICS = ROOT / "experiments_clean_metrics_lfp.html"


def split_traces():
    tmp = TRACES.with_suffix(".html.tmp")
    extracted = False
    with TRACES.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as out, DATA_JS.open(
        "w", encoding="utf-8"
    ) as data_js:
        for line in src:
            if (not extracted) and 'id="trace-json"' in line:
                start = line.find(">") + 1
                end = line.rfind("</script>")
                json_text = line[start:end]
                data_js.write("window.TRACE_DATA=")
                data_js.write(json_text)
                data_js.write(";\n")
                out.write('<script src="experiments_clean_traces_lfp_data.js"></script>\n')
                extracted = True
                print(f"data.js {DATA_JS.stat().st_size/1e6:.1f} MB" if DATA_JS.exists() else "wrote data.js")
                continue
            out.write(line)
    if not extracted:
        tmp.unlink(missing_ok=True)
        raise SystemExit("trace-json not found")
    tmp.replace(TRACES)
    print("split traces html", TRACES.stat().st_size, "data", DATA_JS.stat().st_size)


def patch_traces():
    text = TRACES.read_text(encoding="utf-8")
    text = text.replace(
        "const DATA = JSON.parse(document.getElementById(\"trace-json\").textContent);",
        "const DATA = window.TRACE_DATA;",
    )
    if 'id="loadNote"' not in text:
        text = text.replace(
            "<main>\n  <div class=\"viewbar\">",
            "<main>\n  <p class=\"cap\" id=\"loadNote\">Loading LFP holdout traces…</p>\n  <div class=\"viewbar\">",
        )
    if "loadNote" in text and "loadNote).remove" not in text and "loadNote).style" not in text:
        text = text.replace(
            "window.addEventListener(\"resize\", () => renderCharts());\nrender();",
            """window.addEventListener("resize", () => renderCharts());
(function boot() {
  const note = document.getElementById("loadNote");
  if (!window.TRACE_DATA || !window.TRACE_DATA.evals) {
    if (note) note.textContent = "Failed to load experiments_clean_traces_lfp_data.js. Keep it next to this HTML, or serve the folder with python3 -m http.server.";
    return;
  }
  if (note) note.remove();
  render();
})();""",
        )
    text = text.replace(
        """function availableFids() {
  const m = foldMap();
  if (m) return Object.keys(m);
  const p = evalObj();
  return p ? [p.fid] : [];
}""",
        """function availableFids() {
  const m = foldMap();
  const ids = m ? Object.keys(m) : (evalObj() && evalObj().fid ? [evalObj().fid] : []);
  ids.sort((a, b) => {
    const ma = /^lto(\\d+)$/.exec(a), mb = /^lto(\\d+)$/.exec(b);
    if (ma && mb) return Number(ma[1]) - Number(mb[1]);
    return a.localeCompare(b);
  });
  return ids;
}""",
    )
    text = text.replace(
        """    EVALS.filter((e) => DATA.evals[e]).map((e) => ({
      id: e,
      label: e,
    })),""",
        """    EVALS.filter((e) => DATA.evals[e]).map((e) => ({
      id: e,
      label: e + " · " + Object.keys((DATA.evals[e].packs) || {}).length + " folds",
    })),""",
    )
    text = text.replace(
        "Paw speed vs time from stimOn for every scored neural model, with the ground-truth wheel |ω| subplot under each trial. Filter by learned HDBSCAN id or the manual atlas name. Native ",
        "Paw speed vs time from stimOn for spikes, LFP-only, and spikes+LFP. Filter by learned HDBSCAN id, manual atlas name, or shape. Native ",
    )
    text = text.replace(
        "Wheel |ω| subplot is ground truth from the dataset. Longest trial in this fold is ",
        "Longest trial in this fold is ",
    )
    text = text.replace(
        '    .map((m) => ({ m, r2: tr.r[lossKey(m)] }))',
        '    .map((m) => ({ m, r2: tr.r ? tr.r[lossKey(m)] : null }))',
    )
    TRACES.write_text(text, encoding="utf-8")
    print("patched traces html", TRACES.stat().st_size)


def patch_metrics():
    text = METRICS.read_text(encoding="utf-8")
    old = """  document.getElementById("objPills").style.display = "none";
  document.getElementById("lossPills").style.display = "none";
  state.obj = "trace";
  state.loss = "mse";
  const statEl = document.getElementById("statPills");
  if (statEl) {
    statEl.style.display = showLoss ? "" : "none";
    if (showLoss) pill(statEl, [
      {id:"mean", label:"R²: mean"},
      {id:"median", label:"R²: median"},
    ], state.stat, id=>{state.stat=id; render();});
  }"""
    new = """  document.getElementById("objPills").style.display = "none";
  document.getElementById("lossPills").style.display = "none";
  state.obj = "trace";
  state.loss = "mse";
  const showLoss = true;
  const statEl = document.getElementById("statPills");
  if (statEl) {
    statEl.style.display = "";
    pill(statEl, [
      {id:"mean", label:"R²: mean"},
      {id:"median", label:"R²: median"},
    ], state.stat, id=>{state.stat=id; render();});
  }"""
    if old not in text:
        raise SystemExit("metrics showLoss block not found")
    text = text.replace(old, new, 1)

    old_bars = """function drawBars(cv, cats, vals, errHi, y0, errLo) {
  const dpr = Math.max(1, window.devicePixelRatio||1);
  const host = cv.parentElement;
  const w = Math.max(280, host.clientWidth || 800);
  const h = Math.max(140, host.clientHeight || 200);
  cv.style.width = "100%";
  cv.style.height = "100%";
  cv.width = Math.round(w*dpr);
  cv.height = Math.round(h*dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--fill").trim();
  ctx.fillRect(0,0,w,h);
  if (!cats.length) return;
  const eHi = errHi || [];
  const eLo = errLo || eHi;
  const pad = {l:48,r:12,t:10,b:28};
  const iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  let lo = Math.min(0, ...vals.map((v,i)=>v-(eLo[i]||0)));
  let hi = Math.max(0.05, ...vals.map((v,i)=>v+(eHi[i]||0)));
  if (y0 != null) lo = Math.min(lo, y0);
  if (hi - lo < 1e-6) { hi = lo + 0.05; }
  const yAt = v => pad.t + (hi-v)/(hi-lo)*ih;
  const xAt = i => pad.l + (i+0.5)/cats.length*iw;
  ctx.strokeStyle = "#2c3340"; ctx.fillStyle = "#9aa3b2"; ctx.font = "11px ui-sans-serif,system-ui";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let t=0;t<=4;t++) {
    const v = lo + (hi-lo)*t/4; const y = yAt(v);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+iw,y); ctx.stroke();
    ctx.fillText(v.toFixed(2), pad.l-8, y);
  }
  const z = yAt(0);
  ctx.strokeStyle = "#9aa3b2"; ctx.beginPath(); ctx.moveTo(pad.l,z); ctx.lineTo(pad.l+iw,z); ctx.stroke();
  const bw = Math.min(36, iw/cats.length*0.5);
  cats.forEach((c,i) => {
    const x = xAt(i), v = vals[i], hiE = eHi[i]||0, loE = eLo[i]||0;
    const y = yAt(v);
    ctx.fillStyle = "#6ea8fe";
    const top = Math.min(y,z), hh = Math.max(1, Math.abs(y-z));
    ctx.fillRect(x-bw/2, top, bw, hh);
    ctx.strokeStyle = "#e8eaed"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, yAt(v+hiE)); ctx.lineTo(x, yAt(v-loE)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x-5, yAt(v+hiE)); ctx.lineTo(x+5, yAt(v+hiE)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x-5, yAt(v-loE)); ctx.lineTo(x+5, yAt(v-loE)); ctx.stroke();
    ctx.fillStyle = "#9aa3b2"; ctx.textAlign="center"; ctx.textBaseline="top"; ctx.lineWidth=1;
    ctx.fillText(c, x, pad.t+ih+8);
  });
}"""
    new_bars = """function drawBars(cv, cats, vals, errHi, y0, errLo, colors) {
  const dpr = Math.max(1, window.devicePixelRatio||1);
  const host = cv.parentElement;
  const w = Math.max(280, host.clientWidth || 800);
  const h = Math.max(160, host.clientHeight || 200);
  cv.style.width = "100%";
  cv.style.height = "100%";
  cv.width = Math.round(w*dpr);
  cv.height = Math.round(h*dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--fill").trim();
  ctx.fillRect(0,0,w,h);
  if (!cats.length) return;
  const eHi = errHi || [];
  const eLo = errLo || eHi;
  const tilt = cats.length > 6;
  const pad = {l:48,r:12,t:10,b: tilt ? 52 : 28};
  const iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  let lo = Math.min(0, ...vals.map((v,i)=>v-(eLo[i]||0)));
  let hi = Math.max(0.05, ...vals.map((v,i)=>v+(eHi[i]||0)));
  if (y0 != null) lo = Math.min(lo, y0);
  if (hi - lo < 1e-6) { hi = lo + 0.05; }
  const yAt = v => pad.t + (hi-v)/(hi-lo)*ih;
  const xAt = i => pad.l + (i+0.5)/cats.length*iw;
  ctx.strokeStyle = "#2c3340"; ctx.fillStyle = "#9aa3b2"; ctx.font = "11px ui-sans-serif,system-ui";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let t=0;t<=4;t++) {
    const v = lo + (hi-lo)*t/4; const y = yAt(v);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+iw,y); ctx.stroke();
    ctx.fillText(v.toFixed(2), pad.l-8, y);
  }
  const z = yAt(0);
  ctx.strokeStyle = "#9aa3b2"; ctx.beginPath(); ctx.moveTo(pad.l,z); ctx.lineTo(pad.l+iw,z); ctx.stroke();
  const bw = Math.min(36, iw/cats.length*0.5);
  cats.forEach((c,i) => {
    const x = xAt(i), v = vals[i], hiE = eHi[i]||0, loE = eLo[i]||0;
    const y = yAt(v);
    ctx.fillStyle = (colors && colors[i]) || "#6ea8fe";
    const top = Math.min(y,z), hh = Math.max(1, Math.abs(y-z));
    ctx.fillRect(x-bw/2, top, bw, hh);
    ctx.strokeStyle = "#e8eaed"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, yAt(v+hiE)); ctx.lineTo(x, yAt(v-loE)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x-5, yAt(v+hiE)); ctx.lineTo(x+5, yAt(v+hiE)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x-5, yAt(v-loE)); ctx.lineTo(x+5, yAt(v-loE)); ctx.stroke();
    ctx.fillStyle = "#9aa3b2"; ctx.textAlign="center"; ctx.textBaseline="top"; ctx.lineWidth=1;
    if (tilt) {
      ctx.save();
      ctx.translate(x, pad.t+ih+8);
      ctx.rotate(-0.65);
      ctx.textAlign = "right";
      ctx.fillText(c, 0, 0);
      ctx.restore();
    } else {
      ctx.fillText(c, x, pad.t+ih+8);
    }
  });
}"""
    if old_bars not in text:
        raise SystemExit("drawBars not found")
    text = text.replace(old_bars, new_bars, 1)
    text = text.replace(
        "  drawBars(document.getElementById(\"c1\"), cats, v1, eHi, dir ? 0.4 : 0, eLo);",
        "  drawBars(document.getElementById(\"c1\"), cats, v1, eHi, dir ? 0.4 : 0, eLo, chart.map(r=>MCOL[r.m]));",
    )
    text = text.replace(
        "    drawBars(document.getElementById(\"c2\"), cats, v2, e2, 0);",
        "    drawBars(document.getElementById(\"c2\"), cats, v2, e2, 0, e2, chart.map(r=>MCOL[r.m]));",
    )
    # MCOL is defined after render(); move a stub before render by using global which exists at call time.
    METRICS.write_text(text, encoding="utf-8")
    print("patched metrics html", METRICS.stat().st_size)


def main():
    split_traces()
    patch_traces()
    patch_metrics()
    print("done")


if __name__ == "__main__":
    main()
