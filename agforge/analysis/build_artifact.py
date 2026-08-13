"""Build a self-contained interactive review page for the contact-method comparison.

A GIF plays at a fixed rate and cannot be interrogated; the point of this page is that the hit
index is a CONTROL, so the animation and every metric move together under the reviewer's hand.
Frames are embedded as data URIs because the artifact CSP blocks every external host.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image, ImageSequence

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")
BATCH = os.path.join(OUT, "batch17fix")
REAL = os.path.join(OUT, "real_meshes")
DEST = os.path.join(BATCH, "review.html")

ARMS = [
    ("g1_grid_prod", "grid + teleport (production)", "#e8833a"),
    ("g0_grid_alone", "grid alone (no teleport)", "#3aa7e8"),
    ("p1_particle", "particle mode", "#7ac74f"),
]
REAL_COLOR = "#3f6fb0"
MAXHIT = 17


def frames_from(gif, width=1100, quality=68):
    out = []
    for f in ImageSequence.Iterator(Image.open(gif)):
        im = f.copy().convert("RGB")
        h = int(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def main():
    traj = json.load(open(os.path.join(BATCH, "trajectory.json")))

    real_len = {}
    for h in range(1, MAXHIT + 1):
        p = os.path.join(REAL, "hit_%02d.npz" % h)
        if os.path.exists(p):
            with np.load(p) as z:
                real_len[h] = float(np.ptp(z["V_after"][:, 0]))

    plain = frames_from(os.path.join(BATCH, "render", "contact_methods.gif"))
    err = frames_from(os.path.join(BATCH, "render", "contact_methods_error.gif"))
    print("frames: %d plain, %d error" % (len(plain), len(err)))

    series = {}
    for tag, label, color in ARMS:
        t = traj.get(tag, {})
        series[tag] = {
            "label": label, "color": color,
            "iou": [t.get(str(h), {}).get("iou") for h in range(1, MAXHIT + 1)],
            "dev": [t.get(str(h), {}).get("dev_p95") for h in range(1, MAXHIT + 1)],
            "pen": [t.get(str(h), {}).get("pen_max_mm") for h in range(1, MAXHIT + 1)],
            "len": [t.get(str(h), {}).get("span_x") for h in range(1, MAXHIT + 1)],
        }
    data = {
        "arms": [a[0] for a in ARMS],
        "series": series,
        "real_len": [real_len.get(h) for h in range(1, MAXHIT + 1)],
        "maxhit": MAXHIT,
        "realColor": REAL_COLOR,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data)) \
                   .replace("__PLAIN__", json.dumps(plain)) \
                   .replace("__ERR__", json.dumps(err))
    with open(DEST, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s (%.1f MB)" % (DEST, os.path.getsize(DEST) / 1e6))


TEMPLATE = r"""<title>Forge contact methods — hit-by-hit review</title>
<style>
:root{
  --paper:#faf8f5; --ink:#17140f; --muted:#8a8177; --rule:#e2dcd2; --card:#fffdfa;
  --accent:#c2632a;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#14120f; --ink:#ece7df; --muted:#9b9287; --rule:#2e2922; --card:#1b1814;
         --accent:#e8963f; }
}
:root[data-theme="dark"]{ --paper:#14120f; --ink:#ece7df; --muted:#9b9287; --rule:#2e2922;
  --card:#1b1814; --accent:#e8963f; }
:root[data-theme="light"]{ --paper:#faf8f5; --ink:#17140f; --muted:#8a8177; --rule:#e2dcd2;
  --card:#fffdfa; --accent:#c2632a; }

*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5}
.wrap{max-width:1280px;margin:0 auto;padding:0 20px 64px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}

header{position:sticky;top:0;z-index:10;background:var(--paper);
  border-bottom:1px solid var(--rule);padding:14px 0 12px}
h1{margin:0 0 2px;font-size:1.08rem;font-weight:650;letter-spacing:-.01em}
.sub{margin:0;color:var(--muted);font-size:.8rem}

.ctl{display:flex;align-items:center;gap:14px;margin-top:12px;flex-wrap:wrap}
input[type=range]{flex:1;min-width:240px;accent-color:var(--accent);height:22px}
button{font:inherit;font-size:.82rem;padding:5px 13px;border:1px solid var(--rule);
  border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.hit{font-size:.95rem;font-weight:650;min-width:104px}

figure{margin:18px 0 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:8px;background:var(--card)}
figcaption{color:var(--muted);font-size:.76rem;margin-top:7px}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:22px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:13px 14px 8px}
.card h2{margin:0;font-size:.83rem;font-weight:640}
.card p{margin:2px 0 8px;color:var(--muted);font-size:.72rem}
svg{width:100%;height:150px;display:block;overflow:visible}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.73rem;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.readout{margin-top:9px;font-size:.76rem;border-top:1px solid var(--rule);padding-top:7px}
.readout div{display:flex;justify-content:space-between;gap:10px;padding:1px 0}
.note{margin-top:26px;padding:13px 15px;border-left:3px solid var(--accent);
  background:var(--card);border-radius:0 6px 6px 0;font-size:.82rem}
.note b{color:var(--accent)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header>
  <h1>Rigid-MPM contact methods vs the real forge scan</h1>
  <p class="sub">17-hit replay · res 10 · CFL 0.45 · CPIC off · enlarged MPM domain
    (<span class="mono">AGF_MPM_X_PAD_LOWER=1.3</span>)</p>
  <div class="ctl">
    <span class="hit mono" id="hitlab">hit 1 / 17</span>
    <input type="range" id="slider" min="1" max="17" value="1" step="1" aria-label="hit index">
    <button id="play">▶ play</button>
    <button id="mode" aria-pressed="false">colour by error</button>
  </div>
</header>

<figure>
  <img id="frame" alt="Contact-method comparison at the selected hit">
  <figcaption id="cap"></figcaption>
</figure>

<div class="grid" id="charts"></div>

<div class="note">
  <b>Read the elongation chart first.</b> Before the domain fix the sim froze at 76.8&nbsp;mm from
  hit&nbsp;13 and every geometry metric appeared to collapse after it — that was the billet hitting
  the edge of the MPM grid, not physics. With the domain enlarged it keeps elongating, and the
  remaining ~11% shortfall against the real part at hit&nbsp;17 is a genuine modelling gap.
  <br><br>
  <b>Penetration reads exactly 0 for any arm with the teleport on</b>, by construction — that pass
  projects particles to <span class="mono">signed_dist ≥ margin</span>. It only discriminates
  between arms that have the teleport off.
</div>
</div>

<script>
const D = __DATA__, PLAIN = __PLAIN__, ERR = __ERR__;
const N = D.maxhit;
let hit = 1, errMode = false, timer = null;

const $ = id => document.getElementById(id);
const img = $("frame"), slider = $("slider"), hitlab = $("hitlab"), cap = $("cap");

const CHARTS = [
  {k:"len",  t:"Elongation", u:"mm", d:"bar length vs the real scan — higher is closer", real:true},
  {k:"iou",  t:"IoU vs real scan", u:"", d:"voxel overlap with the real billet — higher is better"},
  {k:"dev",  t:"Surface error (p95)", u:"mm", d:"95th-pct distance from real surface — lower is better"},
  {k:"pen",  t:"Max penetration", u:"mm", d:"deepest particle inside the die — 0 means the teleport is on"},
];

function vals(c){
  let out = [];
  for (const a of D.arms) out = out.concat(D.series[a][c.k].filter(v => v != null));
  if (c.real) out = out.concat(D.real_len.filter(v => v != null));
  return out;
}

function buildCharts(){
  $("charts").innerHTML = CHARTS.map(c => {
    const leg = D.arms.map(a =>
      `<span><i style="background:${D.series[a].color}"></i>${D.series[a].label}</span>`).join("")
      + (c.real ? `<span><i style="background:${D.realColor}"></i>real scan</span>` : "");
    const rows = D.arms.map(a =>
      `<div><span>${D.series[a].label}</span><span class="mono" id="v-${c.k}-${a}">—</span></div>`).join("")
      + (c.real ? `<div><span>real scan</span><span class="mono" id="v-${c.k}-real">—</span></div>` : "");
    return `<div class="card"><h2>${c.t}</h2><p>${c.d}</p>
      <svg id="svg-${c.k}" viewBox="0 0 320 150" preserveAspectRatio="none" role="img"
           aria-label="${c.t} by hit"></svg>
      <div class="legend">${leg}</div><div class="readout">${rows}</div></div>`;
  }).join("");
  CHARTS.forEach(drawChart);
}

function drawChart(c){
  const v = vals(c);
  if (!v.length) return;
  let lo = Math.min(...v), hi = Math.max(...v);
  if (hi === lo) { hi = lo + 1; }
  const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const W = 320, H = 150, L = 34, B = 18;
  const x = i => L + (W - L - 4) * i / (N - 1);
  const y = val => (H - B) - (H - B - 6) * (val - lo) / (hi - lo);

  let s = "";
  for (let g = 0; g <= 2; g++){
    const yy = 6 + (H - B - 6) * g / 2, val = hi - (hi - lo) * g / 2;
    s += `<line x1="${L}" y1="${yy}" x2="${W-4}" y2="${yy}" stroke="var(--rule)" stroke-width="1"/>`;
    s += `<text x="0" y="${yy+3}" font-size="8" fill="var(--muted)"
           font-family="ui-monospace,monospace">${val.toFixed(val>20?0:2)}</text>`;
  }
  const line = (arr, col) => {
    let d = "", started = false;
    arr.forEach((val, i) => {
      if (val == null) { started = false; return; }
      d += (started ? "L" : "M") + x(i).toFixed(1) + " " + y(val).toFixed(1) + " ";
      started = true;
    });
    return d ? `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"
                 stroke-linejoin="round" stroke-linecap="round"/>` : "";
  };
  if (c.real) s += line(D.real_len, D.realColor);
  for (const a of D.arms) s += line(D.series[a][c.k], D.series[a].color);
  s += `<line id="cur-${c.k}" x1="0" y1="2" x2="0" y2="${H-B}" stroke="var(--accent)"
         stroke-width="1.5" stroke-dasharray="3 3"/>`;
  s += `<text x="${L}" y="${H-5}" font-size="8" fill="var(--muted)"
         font-family="ui-monospace,monospace">hit 1</text>`;
  s += `<text x="${W-26}" y="${H-5}" font-size="8" fill="var(--muted)"
         font-family="ui-monospace,monospace">${N}</text>`;
  document.getElementById("svg-" + c.k).innerHTML = s;
}

function render(){
  const i = hit - 1;
  const src = (errMode ? ERR : PLAIN)[Math.min(i, (errMode?ERR:PLAIN).length - 1)];
  img.src = "data:image/jpeg;base64," + src;
  hitlab.textContent = `hit ${hit} / ${N}`;
  slider.value = hit;
  cap.textContent = errMode
    ? "Sim particles coloured by signed distance to the real surface: red = material where the real part has none (over-fill); blue = inside the real body. Rows: two isometric views, then a longitudinal cross-section."
    : "Columns: real scan, then each contact method. Rows: two isometric views, then a longitudinal cross-section through the bar axis.";

  const W = 320, L = 34;
  const x = idx => L + (W - L - 4) * idx / (N - 1);
  CHARTS.forEach(c => {
    const ln = document.getElementById("cur-" + c.k);
    if (ln){ ln.setAttribute("x1", x(i)); ln.setAttribute("x2", x(i)); }
    for (const a of D.arms){
      const el = document.getElementById(`v-${c.k}-${a}`);
      const val = D.series[a][c.k][i];
      if (el) el.textContent = val == null ? "—" : val.toFixed(c.k === "iou" ? 4 : 2) + " " + c.u;
    }
    if (c.real){
      const el = document.getElementById(`v-${c.k}-real`);
      const val = D.real_len[i];
      if (el) el.textContent = val == null ? "—" : val.toFixed(2) + " mm";
    }
  });
}

slider.addEventListener("input", e => { hit = +e.target.value; render(); });
$("mode").addEventListener("click", e => {
  errMode = !errMode;
  e.target.setAttribute("aria-pressed", String(errMode));
  e.target.textContent = errMode ? "plain colours" : "colour by error";
  render();
});
$("play").addEventListener("click", e => {
  if (timer){ clearInterval(timer); timer = null; e.target.textContent = "▶ play"; return; }
  e.target.textContent = "❚❚ pause";
  timer = setInterval(() => { hit = hit >= N ? 1 : hit + 1; render(); }, 650);
});
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight"){ hit = Math.min(N, hit + 1); render(); }
  if (e.key === "ArrowLeft"){ hit = Math.max(1, hit - 1); render(); }
});

buildCharts();
render();
</script>
"""

if __name__ == "__main__":
    main()
