"use strict";
/* ============================================================
   Narrated explainer video. Every frame is a pure function of the
   global time T (seconds), so frames can be rendered in any order.
   Cue times come from the voice-over's word timestamps (TIMING).
   ============================================================ */
const FPS = 30, LEAD = 0.8, AUDIO_DUR = 155.873, TAIL = 3.4;
const TOTAL = LEAD + AUDIO_DUR + TAIL;
const TOK = TIMING;
const nrm = s => s.toLowerCase().replace(/’/g, "'").replace(/[^a-z0-9']/g, "");
function cue(phrase) {
  const want = phrase.split(/\s+/).map(nrm).filter(Boolean);
  outer: for (let i = 0; i < TOK.length; i++) {
    for (let k = 0; k < want.length; k++) { const t = TOK[i + k]; if (!t || nrm(t.t) !== want[k]) continue outer; }
    return TOK[i].s + LEAD;
  }
  throw new Error("cue not found: " + phrase);
}
const pstart = k => TOK.find(t => t.para === k).s + LEAD;
const E = x => (x <= 0 ? 0 : x >= 1 ? 1 : x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const ap = (T, t0, d = 0.6) => E((T - t0) / d);
function seg(T, pts) {
  if (T <= pts[0][0]) return pts[0][1];
  for (let k = 1; k < pts.length; k++) if (T <= pts[k][0]) {
    const [t0, v0] = pts[k - 1], [t1, v1] = pts[k];
    return lerp(v0, v1, E((T - t0) / (t1 - t0)));
  }
  return pts[pts.length - 1][1];
}
const stage = document.getElementById("stage");
readPalette();
const W10 = weights(10).W, WC10 = weights(10, { type: "consensus" }).W;
function box(parent, cls, x, y, w, html) {
  const e = document.createElement("div");
  e.style.position = "absolute";
  if (cls) e.className = cls;
  e.style.left = x + "px"; e.style.top = y + "px";
  if (w != null) e.style.width = w + "px";
  if (html != null) e.innerHTML = html;
  parent.appendChild(e);
  return e;
}
function fade(e, a, dy = 16) {
  e.style.opacity = a.toFixed(3);
  e.style.transform = a >= 1 ? "none" : `translateY(${((1 - a) * dy).toFixed(1)}px)`;
}
const EQS = [];
function eq(parent, x, y, tex, size = 44) {
  const e = box(parent, "eq", x, y);
  e.style.fontSize = size + "px";
  e.dataset.tex = tex;
  EQS.push(e);
  return e;
}
function svgBox(parent, x, y, w, h) {
  const host = box(parent, "", x, y, w);
  const s = S("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, class: "chart" }, host);
  return s;
}
function header(root, eyebrow, title, start) {
  const eb = box(root, "eyebrow", 110, 120, 820, eyebrow);
  const ti = box(root, "title", 110, 156, 770, title);
  return T => { const a = ap(T, start + 0.1, 0.7); fade(eb, a); fade(ti, a, 24); };
}
function matrixHost(root, x, y, o) {
  const w = (o.left ?? 30) + o.n * o.cell + (o.right ?? 4);
  const host = box(root, "", x, y, w);
  const mx = MatrixView(host, o);
  mx.svg.setAttribute("width", w);
  mx.host = host;
  return mx;
}
const mixRGB = (a, b, t) => rgbStr(lab2rgb(mixLab(rgb2lab(a), rgb2lab(b), t)));

/* ---------- dot table (canvas) ---------- */
function DotTable(parent, x, y, size) {
  const cv = document.createElement("canvas");
  cv.width = size; cv.height = size;
  Object.assign(cv.style, { left: x + "px", top: y + "px", width: size + "px", height: size + "px", transformOrigin: "50% 50%" });
  parent.appendChild(cv);
  const ctx = cv.getContext("2d");
  const L = 62, Tp = 62, g = size - L - 8, c = g / 11;
  const R = rng(11);
  const gauss = () => { let u = 0, v = 0; while (!u) u = R(); while (!v) v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const dots = [];
  for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) {
    const cnt = SIM[i][j], side = Math.ceil(Math.sqrt(cnt));
    const inner = c * 0.84, sp = inner / Math.max(side, 1), off = (c - inner) / 2 + sp / 2;
    for (let k = 0; k < cnt; k++) {
      const r = Math.floor(k / side), q = k % side;
      dots.push({ camp: i > j ? "x" : i < j ? "y" : "d", tx: L + j * c + off + q * sp, ty: Tp + i * c + off + r * sp,
        fx: L + g * 0.5 + gauss() * g * 0.2, fy: Tp + g * 0.45 + gauss() * g * 0.2, delay: R(), size: Math.max(2.2, Math.min(4.2, sp * 0.8)) });
    }
  }
  const inRegion = { diag: (i, j) => i === j, lower: (i, j) => i > j, upper: (i, j) => i < j };
  function outline(fn, color, alpha) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.lineCap = "square"; ctx.beginPath();
    for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) {
      if (!fn(i, j)) continue;
      const x0 = L + j * c, y0 = Tp + i * c, x1 = x0 + c, y1 = y0 + c;
      if (i === 0 || !fn(i - 1, j)) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); }
      if (i === 10 || !fn(i + 1, j)) { ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); }
      if (j === 0 || !fn(i, j - 1)) { ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); }
      if (j === 10 || !fn(i, j + 1)) { ctx.moveTo(x1, y0); ctx.lineTo(x1, y1); }
    }
    ctx.stroke(); ctx.restore();
  }
  function draw(s) {
    ctx.clearRect(0, 0, size, size);
    const col = { x: C["camp-x"], y: C["camp-y"], d: C.diag };
    for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) {
      ctx.fillStyle = C.sheet; ctx.fillRect(L + j * c + 1, Tp + i * c + 1, c - 2, c - 2);
      ctx.fillStyle = C.mid; ctx.fillRect(L + j * c + 1, Tp + i * c + 1, c - 2, c - 2);
      if (s.tint > 0 && W10[i][j] > 0) {
        ctx.globalAlpha = s.tint * 0.55; ctx.fillStyle = ramp("pos", W10[i][j] * 0.75);
        ctx.fillRect(L + j * c + 1, Tp + i * c + 1, c - 2, c - 2); ctx.globalAlpha = 1;
      }
      for (const [k, cc] of [["diag", "d"], ["lower", "x"], ["upper", "y"]]) {
        const a = (s.regions || {})[k] || 0;
        if (a > 0 && inRegion[k](i, j)) { ctx.globalAlpha = a * 0.16; ctx.fillStyle = col[cc]; ctx.fillRect(L + j * c + 1, Tp + i * c + 1, c - 2, c - 2); ctx.globalAlpha = 1; }
      }
    }
    ctx.fillStyle = C.muted; ctx.font = `500 17px "IBM Plex Mono", monospace`;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let k = 0; k < 11; k++) ctx.fillText(String(k), L - 8, Tp + k * c + c / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    for (let k = 0; k < 11; k++) ctx.fillText(String(k), L + k * c + c / 2, Tp - 10);
    ctx.fillStyle = C["ink-2"]; ctx.font = `600 20px Archivo, sans-serif`;
    ctx.fillText(s.colTitle + "  →", L + 5.5 * c, 22);
    ctx.save(); ctx.translate(20, Tp + 5.5 * c); ctx.rotate(-Math.PI / 2); ctx.fillText("←  " + s.rowTitle, 0, 0); ctx.restore();
    const da = s.dotAlpha ?? 1;
    if (da > 0) {
      ctx.globalAlpha = da;
      for (const camp of ["d", "x", "y"]) {
        ctx.fillStyle = col[camp];
        for (const d of dots) {
          if (d.camp !== camp) continue;
          const t = E(clamp((s.fall * 1.5 - d.delay) / 0.5, 0, 1));
          ctx.fillRect(lerp(d.fx, d.tx, t) - d.size / 2, lerp(d.fy, d.ty, t) - d.size / 2, d.size, d.size);
        }
      }
      ctx.globalAlpha = 1;
    }
    if ((s.regions || {}).diag) outline(inRegion.diag, C["ink-2"], s.regions.diag);
    if ((s.regions || {}).lower) outline(inRegion.lower, C["camp-x"], s.regions.lower);
    if ((s.regions || {}).upper) outline(inRegion.upper, C["camp-y"], s.regions.upper);
  }
  return { cv, draw };
}

/* ---------- small chart helpers ---------- */
function hbars(svg, rows, { max, labelW = 120, w, rowH = 56, fmt = v => f(v, 3) }) {
  svg.replaceChildren();
  const X = axisLinear(0, max, labelW, w - 120);
  S("line", { x1: labelW, x2: labelW, y1: 0, y2: rows.length * rowH, stroke: C.rule, "stroke-width": 2 }, svg);
  rows.forEach((r, k) => {
    const y = k * rowH;
    const lab = S("text", { x: labelW - 14, y: y + rowH / 2, "text-anchor": "end", "dominant-baseline": "central", "font-size": 26, "font-family": "Archivo, sans-serif", "font-weight": 650, fill: C["ink-2"] }, svg);
    svgSub(lab, r.label);
    lab.style.opacity = r.a;
    const v = r.value * r.grow;
    S("rect", { x: labelW, y: y + 9, width: Math.max(0, X(v) - labelW), height: rowH - 18, rx: 4, fill: r.color, opacity: r.a }, svg);
    if (r.lo != null && r.grow >= 1) {
      S("line", { x1: X(r.lo), x2: X(r.hi), y1: y + rowH / 2, y2: y + rowH / 2, stroke: C.ink, "stroke-width": 2.5, opacity: r.ci }, svg);
      [r.lo, r.hi].forEach(b => S("line", { x1: X(b), x2: X(b), y1: y + rowH / 2 - 9, y2: y + rowH / 2 + 9, stroke: C.ink, "stroke-width": 2.5, opacity: r.ci }, svg));
    }
    const t = S("text", { x: X(r.hi ?? v) + 14, y: y + rowH / 2, "dominant-baseline": "central", "font-size": 26, opacity: r.a * clamp(r.grow * 1.5 - 0.5, 0, 1) }, svg);
    t.textContent = fmt(r.value);
  });
}
function dotci(svg, rows, { d0, d1, w, labelW = 80, rowH = 62, a = 1 }) {
  svg.replaceChildren();
  const X = axisLinear(d0, d1, labelW, w - 20);
  const h = rows.length * rowH;
  [5.4, 5.6, 5.8, 6.0].forEach(v => {
    S("line", { x1: X(v), x2: X(v), y1: 0, y2: h, stroke: C.grid, "stroke-width": 1.5 }, svg);
    S("text", { x: X(v), y: h + 22, class: "tick", "text-anchor": "middle" }, svg).textContent = f(v, 1);
  });
  rows.forEach((r, k) => {
    const y = k * rowH + rowH / 2;
    const lab = S("text", { x: labelW - 14, y, "text-anchor": "end", "dominant-baseline": "central", "font-size": 26, "font-family": "Archivo, sans-serif", "font-weight": 650, fill: C["ink-2"] }, svg);
    svgSub(lab, r.label);
    const g = S("g", { opacity: r.a }, svg);
    S("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: r.color, "stroke-width": 5, "stroke-linecap": "round", opacity: r.ci }, g);
    S("circle", { cx: X(r.value), cy: y, r: 10, fill: r.color, stroke: C.sheet, "stroke-width": 3 }, g);
    S("text", { x: X(r.hi) + 14, y, "dominant-baseline": "central", "font-size": 24, opacity: r.ci }, g).textContent = f(r.value, 2);
  });
}
function svgSub(el, label) {
  el.replaceChildren();
  const m = String(label).match(/^(.*?)_(\w+)(.*)$/);
  if (!m) { el.textContent = label; return; }
  el.appendChild(document.createTextNode(m[1]));
  const t = S("tspan", { "baseline-shift": "sub", "font-size": "0.72em" }, el); t.textContent = m[2];
  if (m[3]) el.appendChild(document.createTextNode(m[3]));
}

/* ============================================================
   Scenes
   ============================================================ */
const SCENES = [];

/* 0 · title */
(function () {
  const root = box(stage, "scene", 0, 0, 1920);
  const eb = box(root, "eyebrow", 110, 300, 860, "An animated guide · bipolarization");
  const ti = box(root, "big-title", 104, 340, 900, "Two camps,<br>one table");
  const le = box(root, "lede", 110, 700, 790, "How the <b>bipolarization</b> method turns the answers to two survey items into polarization and consensus scores.");
  const dt = DotTable(root, 1040, 130, 770);
  let drawn = false;
  SCENES.push({ root, start: 0, update(T) {
    fade(eb, ap(T, 0.1, 0.7)); fade(ti, ap(T, 0.2, 0.8), 30); fade(le, ap(T, 0.7, 0.8));
    const a = ap(T, 0.2, 1.2);
    dt.cv.style.opacity = a.toFixed(3);
    dt.cv.style.transform = `scale(${(0.96 + 0.04 * a + 0.004 * Math.max(0, T - 1.4)).toFixed(4)})`;
    if (!drawn) { dt.draw({ fall: 1, tint: 1, rowTitle: "rating of X  (i)", colTitle: "rating of Y  (j)" }); drawn = true; }
  } });
})();

/* 1 · the problem */
(function () {
  const st = pstart(1) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "1 · Why a joint table", "Marginals can't see camps", st);
  const c1 = cue("Most polarization measures"), c2 = cue("polarization is about pairs"), c3 = cue("warm to one party"),
    c4 = cue("identical answer distributions"), c5 = cue("completely opposite camps");
  const m = [0.37, 0.03, 0.03, 0.04, 0.04, 0.10, 0.05, 0.06, 0.07, 0.04, 0.17];
  const K = 11, F = [];
  m.reduce((s, v, k) => (F[k] = s + v), 0);
  const Fm = k => (k < 0 ? 0 : F[k]);
  const ind = zeros(K), co = zeros(K), ctr = zeros(K);
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    ind[i][j] = m[i] * m[j]; co[i][j] = i === j ? m[i] : 0;
    ctr[i][j] = Math.max(0, Math.min(Fm(i), 1 - Fm(j - 1)) - Math.max(Fm(i - 1), 1 - Fm(j)));
  }
  const joint = t => ind.map((r, i) => r.map((v, j) => (t >= 0 ? (1 - t) * v + t * co[i][j] : (1 + t) * v - t * ctr[i][j])));
  function marg(y, name) {
    const wrap = box(root, "", 110, y, 720);
    box(wrap, "label", 0, 0, 720, `${name}<span class="mono" style="font-size:21px;color:var(--muted);margin-left:16px;font-weight:400">mean 4.21 · SD 3.92</span>`);
    const s = svgBox(wrap, 0, 44, 640, 120);
    const X = axisLinear(0, 11, 10, 630);
    m.forEach((v, k) => {
      S("rect", { x: X(k) + 5, y: 96 - v / 0.37 * 86, width: X(1) - X(0) - 10, height: v / 0.37 * 86, rx: 3, fill: C.diag }, s);
      S("text", { x: X(k) + (X(1) - X(0)) / 2, y: 116, class: "tick", "text-anchor": "middle" }, s).textContent = k;
    });
    const badge = box(wrap, "", 0, 36, 652);
    Object.assign(badge.style, { height: "136px", border: `3px solid ${C.pen}`, borderRadius: "10px", left: "-6px" });
    const tag = box(wrap, "label", 476, -4, null, "identical");
    Object.assign(tag.style, { color: C.sheet, background: C.pen, padding: "2px 14px", borderRadius: "6px", fontSize: "22px" });
    return { wrap, badge, tag };
  }
  const mA = marg(372, "Answers to item X"), mB = marg(576, "Answers to item Y");
  const ro = box(root, "readout", 110, 790, 420, `<div class="k">overall score S<sub>pol</sub></div><div class="v">0.000</div>`);
  const roV = ro.querySelector(".v");
  const mx = matrixHost(root, 1010, 120, { n: 11, cell: 54, left: 104, top: 104, gap: 2, fs: 12, rowTitle: "rating of X  (i)", colTitle: "rating of Y  (j)", rowTitleX: 14, colTitleY: 16 });
  const gM = S("g", {}, mx.svg);
  m.forEach((v, k) => {
    const len = v / 0.37 * 58;
    S("rect", { x: mx.L - 30 - len, y: mx.T + k * mx.c + 6, width: len, height: mx.c - 12, rx: 2, fill: C.diag }, gM);
    S("rect", { x: mx.L + k * mx.c + 6, y: mx.T - 26 - len, width: mx.c - 12, height: len, rx: 2, fill: C.diag }, gM);
  });
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    fade(mA.wrap, ap(T, c1, 0.6)); fade(mB.wrap, ap(T, c1 + 0.25, 0.6));
    const id = ap(T, c4, 0.5) * (1 - ap(T, c5 + 1.4, 0.6));
    [mA, mB].forEach(o => { o.badge.style.opacity = id; o.tag.style.opacity = id; });
    fade(mx.host, ap(T, c2, 0.7));
    fade(ro, ap(T, c2 + 0.3, 0.6));
    const t = seg(T, [[c3, 0], [c3 + 1.6, -1], [c4, -1], [c4 + 1.5, 1], [c5, 1], [c5 + 1.7, -1]]);
    const P = joint(t);
    mx.set(P, { fill: v => ramp("mass", Math.sqrt(v / 0.37)) });
    mx.guide(t > 0.5 ? "diag" : t < -0.5 ? "anti" : null);
    roV.textContent = f(polScores(P, W10).Spol, 3);
  } });
})();

/* 2 · the joint table */
(function () {
  const st = pstart(2) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "2 · The joint table", "Tally the pairs into one table", st);
  const cA = cue("Each respondent rates"), cB = cue("eleven by eleven"), cD = cue("The diagonal holds"), cL = cue("Below it sit"), cU = cue("above it those");
  const dt = DotTable(root, 990, 110, 790);
  const e1 = eq(root, 110, 400, String.raw`a_{ij}=\#\{k:\ x_k=i,\ y_k=j\}`, 46);
  const e2 = eq(root, 110, 492, String.raw`(N+1)\times(N+1)=11\times 11`, 46);
  const l1 = box(root, "legend-row", 110, 610, 760, `<i style="background:${C.diag}"></i>i = j <span>rated both the same</span>`);
  const l2 = box(root, "legend-row", 110, 676, 760, `<i style="background:${C["camp-x"]}"></i>i &gt; j <span>prefers X · lower triangle</span>`);
  const l3 = box(root, "legend-row", 110, 742, 760, `<i style="background:${C["camp-y"]}"></i>i &lt; j <span>prefers Y · upper triangle</span>`);
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    fade(e1, ap(T, cA, 0.6)); fade(e2, ap(T, cB - 0.2, 0.6));
    fade(l1, ap(T, cD, 0.5)); fade(l2, ap(T, cL, 0.5)); fade(l3, ap(T, cU, 0.5));
    dt.draw({ fall: clamp((T - cA) / 4.4, 0, 1), dotAlpha: ap(T, cA - 0.2, 0.5), tint: 0,
      regions: { diag: ap(T, cD, 0.5), lower: ap(T, cL, 0.5), upper: ap(T, cU, 0.5) },
      rowTitle: "rating of X  (i)", colTitle: "rating of Y  (j)" });
  } });
})();

/* 3 · weights */
(function () {
  const st = pstart(3) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "3 · The weight matrix · Eq. 8–12", "Weigh each cell by distance and position", st);
  const cDist = cue("Distance how far"), cAgr = cue("And agreement"), cMul = cue("Multiply them and the weight"), cCor = cue("two far corners"), cGap = cue("A five point gap");
  const N = 10, D = [], A = [], Wm = [];
  for (let i = 0; i <= N; i++) { D.push([]); A.push([]); Wm.push([]); for (let j = 0; j <= N; j++) { const d = Math.abs(i - j) / N, a = 1 - Math.abs(i + j - N) / N; D[i].push(d); A[i].push(a); Wm[i].push(d * a); } }
  const xs = [880, 1205, 1530], caps = ["distance d", "agreement a", "weight w = d · a"];
  const views = xs.map((x, k) => {
    const cap = box(root, "cap", x, 196, 288, caps[k]);
    const mv = matrixHost(root, x, 236, { n: 11, cell: 24, left: 20, top: 20, gap: 1.2, ticks: false });
    return { cap, mv };
  });
  const ops = [box(root, "op", 1166, 330, null, "×"), box(root, "op", 1491, 330, null, "=")];
  const e1 = eq(root, 110, 400, String.raw`d_{ij}=\frac{|i-j|}{N}`, 48);
  const e2 = eq(root, 110, 506, String.raw`a_{ij}=1-\frac{|(i+j)-N|}{N}`, 48);
  const e3 = eq(root, 110, 618, String.raw`w_{ij}=d_{ij}\cdot a_{ij}`, 48);
  const n1 = box(root, "note", 110, 706, 720, "Largest in the two far corners, <b>(0, N)</b> and <b>(N, 0)</b>.");
  const card = box(root, "card", 880, 560, 930);
  card.style.padding = "22px 30px";
  card.innerHTML = `<div class="label" style="margin-bottom:12px">Three cells, each five points apart</div>
    <div class="mono" style="font-size:29px;line-height:1.65;color:var(--ink-2)">(0, 5) &nbsp;d = 0.5 &nbsp;a = 0.5 &nbsp;→ w = <b style="color:var(--ink)">0.25</b><br>
    <span style="color:var(--ink)">(3, 8) &nbsp;d = 0.5 &nbsp;a = 0.9 &nbsp;→ w = <b style="color:var(--pos-3)">0.45</b></span><br>
    (5, 10) d = 0.5 &nbsp;a = 0.5 &nbsp;→ w = <b style="color:var(--ink)">0.25</b></div>
    <div class="note" style="margin-top:10px;font-size:26px">Straddling the midpoint: <b>0.45 / 0.25 = 1.8×</b> the weight.</div>`;
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    const aD = ap(T, cDist - 0.1, 0.6), aA = ap(T, cAgr - 0.1, 0.6), aW = ap(T, cMul - 0.1, 0.6);
    fade(views[0].cap, aD); fade(views[0].mv.host, aD);
    fade(views[1].cap, aA); fade(views[1].mv.host, aA); fade(ops[0], aA);
    fade(views[2].cap, aW); fade(views[2].mv.host, aW); fade(ops[1], aW);
    fade(e1, aD); fade(e2, aA); fade(e3, aW); fade(n1, ap(T, cCor, 0.5)); fade(card, ap(T, cGap, 0.6));
    views[0].mv.set(D, { fill: v => ramp("pos", v) });
    views[1].mv.set(A, { fill: v => ramp("mass", v * 0.8) }); views[1].mv.guide("anti");
    views[2].mv.set(Wm, { fill: v => ramp("pos", v) }); views[2].mv.guide("anti");
    const pings = T >= cGap ? [[0, 5], [3, 8], [5, 10]] : T >= cCor ? [[0, 10], [10, 0]] : [];
    views[2].mv.ping(pings);
  } });
})();

/* 4 · kernels */
(function () {
  const st = pstart(4) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "4 · Kernels · Eq. 12–15", "Bend the distance curve", st);
  const k0 = pstart(4), cR = cue("final rescaling");
  const e1 = eq(root, 110, 400, String.raw`m_{ij}=d_{ij}^{\,p}\,a_{ij}^{\,q}`, 48);
  const e2 = eq(root, 110, 500, String.raw`g_{ij}=\bigl(1-e^{-d_{ij}^2/2\sigma^2}\bigr)\operatorname{sign}(d_{ij})`, 42);
  const e3 = eq(root, 110, 612, String.raw`w_{ij}=\frac{m_{ij}}{\max|m|}\in[-1,1]`, 48);
  const cv = svgBox(root, 880, 190, 640, 520);
  const mv = matrixHost(root, 1536, 250, { n: 11, cell: 24, left: 4, top: 4, gap: 1.2, ticks: false });
  const mcap = box(root, "cap", 1536, 520, 272, "normalised W");
  const X = axisLinear(0, 1, 70, 620), Y = axisLinear(0, 1, 440, 40);
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    const gaussian = T >= k0 + 3.3;
    const p = seg(T, [[k0 + 0.2, 1], [k0 + 1.5, 0.45], [k0 + 3.0, 2.6]]);
    const sg = seg(T, [[k0 + 3.4, 0.2], [k0 + 5.8, 0.55]]);
    fade(e1, ap(T, st + 0.3, 0.6)); fade(e2, ap(T, k0 + 3.2, 0.6)); fade(e3, ap(T, cR, 0.6));
    const fa = ap(T, st + 0.4, 0.6); fade(cv.parentElement, fa); fade(mv.host, fa); fade(mcap, fa);
    const fd = d => (gaussian ? 1 - Math.exp(-(d * d) / (2 * sg * sg)) : Math.pow(d, p));
    const mx = fd(1);
    cv.replaceChildren();
    [0, 0.25, 0.5, 0.75, 1].forEach(v => {
      S("line", { x1: X(0), x2: X(1), y1: Y(v), y2: Y(v), stroke: C.grid, "stroke-width": 1.5 }, cv);
      S("text", { x: X(0) - 12, y: Y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "central", "font-size": 18 }, cv).textContent = v;
      S("text", { x: X(v), y: Y(0) + 28, class: "tick", "text-anchor": "middle", "font-size": 18 }, cv).textContent = v;
    });
    S("text", { x: X(0.5), y: 506, class: "axt", "text-anchor": "middle", "font-size": 21 }, cv).textContent = "distance d";
    let raw = "", norm = "";
    for (let k = 0; k <= 120; k++) { const d = k / 120, v = fd(d); raw += (k ? "L" : "M") + X(d).toFixed(1) + " " + Y(v).toFixed(1); norm += (k ? "L" : "M") + X(d).toFixed(1) + " " + Y(v / mx).toFixed(1); }
    if (gaussian) {
      S("line", { x1: X(sg), x2: X(sg), y1: Y(0), y2: Y(1), stroke: C.muted, "stroke-dasharray": "4 5", "stroke-width": 1.5 }, cv);
      S("text", { x: X(sg) + 8, y: Y(0.06), "font-size": 22, fill: C.muted }, cv).textContent = "σ";
      S("path", { d: raw, fill: "none", stroke: C.muted, "stroke-width": 2.5, "stroke-dasharray": "8 6", opacity: ap(T, cR, 0.6) }, cv);
    }
    S("path", { d: norm, fill: "none", stroke: C["pos-2"], "stroke-width": 4 }, cv);
    for (let k = 0; k <= 10; k++) S("circle", { cx: X(k / 10), cy: Y(fd(k / 10) / mx), r: 6, fill: C["pos-2"], stroke: C.sheet, "stroke-width": 2.5 }, cv);
    const lab = S("text", { x: X(0) + 8, y: Y(1) - 8, "font-size": 26, "font-family": "Archivo, sans-serif", "font-weight": 650, fill: C.ink }, cv);
    lab.textContent = gaussian ? `Gaussian kernel · σ = ${f(sg, 2)}` : `power kernel · p = ${f(p, 2)}`;
    if (gaussian) {
      const t2 = S("text", { x: X(0) + 8, y: Y(1) + 24, "font-size": 21, fill: C["ink-2"], opacity: ap(T, cR, 0.6) }, cv);
      t2.textContent = `raw maximum ${f(mx, 3)} → rescaled to 1`;
    }
    const opts = gaussian ? { kernel: "gaussian", sigma: sg, q: 1 } : { kernel: "power", p, q: 1 };
    mv.set(weights(10, opts).W, { fill: v => ramp("pos", v) });
  } });
})();

/* 5 · scores */
(function () {
  const st = pstart(5) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "5 · Scores · Eq. 16–20", "Multiply, then sum each triangle", st);
  const cDiv = cue("divide every count"), cMul = cue("Multiply it cell by cell"), cAdd = cue("add up each triangle"),
    cX = cue("one score for the camp that prefers X"), cY = cue("one for the camp that prefers Y"), cSum = cue("and their sum"),
    cAKP = cue("For the AKP and CHP pair"), cCor = cue("two extreme corners");
  const n = total(SIM), P = SIM.map(r => r.map(v => v / n)), WPm = P.map((r, i) => r.map((v, j) => v * W10[i][j]));
  const pt = polScores(SIM, W10);
  const mx = matrixHost(root, 1060, 112, { n: 11, cell: 60, left: 66, top: 66, gap: 2, fs: 15, rowTitle: "X  (i)", colTitle: "Y  (j)", rowTitleX: 14, colTitleY: 18 });
  const [axRow, axCol] = mx.svg.querySelectorAll(".axt");
  mx.regions([{ cells: (i, j) => i > j, color: C["camp-x"] }, { cells: (i, j) => i < j, color: C["camp-y"] }]);
  const regs = mx.svg.querySelectorAll(".region");
  const e1 = eq(root, 110, 384, String.raw`p_{ij}=\frac{a_{ij}}{n}`, 44);
  const e2 = eq(root, 420, 392, String.raw`\mathbf{WP}=\mathbf{W}\circ\mathbf{P}`, 44);
  const e3 = eq(root, 110, 488, String.raw`S_X=\sum_{i>j}\mathrm{WP}_{ij}\qquad S_Y=\sum_{i<j}\mathrm{WP}_{ij}`, 42);
  const bars = svgBox(root, 110, 610, 780, 170);
  const cap = box(root, "", 1060, 34, 720, `<div class="label" style="color:var(--ink)">AKP × CHP · Turkey General Social Survey 2024 · n = 2,218</div><div class="note" style="font-size:20px;margin-top:2px">Cell counts simulated to reproduce the paper's published Figure 2(c).</div>`);
  const call = box(root, "note", 110, 806, 800, `Two corners: <b>0.140 + 0.117</b> = 57% of S<sub>pol</sub>, from 26% of respondents.`);
  const massC = (i, j) => rampRGB(C.lab.mass, Math.sqrt(SIM[i][j] / 311));
  const wpC = (i, j) => rampRGB(C.lab.pos, WPm[i][j] / 0.14);
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    fade(mx.host, ap(T, st + 0.2, 0.7));
    fade(e1, ap(T, cDiv, 0.6)); fade(e2, ap(T, cMul, 0.6)); fade(e3, ap(T, cAdd, 0.6));
    fade(bars.parentElement, ap(T, cX - 0.2, 0.5)); fade(cap, ap(T, cAKP, 0.6)); fade(call, ap(T, cCor + 0.3, 0.6));
    const pr = T >= cDiv + 0.5, eW = ap(T, cMul + 0.2, 1.0);
    mx.set(SIM, {
      fill: (v, i, j) => (v ? (eW > 0 ? mixRGB(massC(i, j), wpC(i, j), eW) : rgbStr(massC(i, j))) : C.sheet),
      text: (v, i, j) => (!v ? "" : eW >= 0.5 ? fc(WPm[i][j], 3) : pr ? fc(P[i][j], 3) : String(v)),
    });
    mx.guide(eW > 0.5 ? "diag" : null);
    regs.forEach(r => { r.style.opacity = ap(T, cAdd, 0.6); });
    mx.ping(T >= cCor ? [[0, 10], [10, 0]] : []);
    const akp = ap(T, cAKP, 0.6) >= 0.5;
    axRow.textContent = akp ? "AKP affinity  (i)" : "rating of X  (i)";
    axCol.textContent = akp ? "CHP affinity  (j)" : "rating of Y  (j)";
    hbars(bars, [
      { label: "S_X", value: pt.SX, color: C["camp-x"], a: ap(T, cX, 0.4), grow: ap(T, cX + 0.2, 0.9) },
      { label: "S_Y", value: pt.SY, color: C["camp-y"], a: ap(T, cY, 0.4), grow: ap(T, cY + 0.2, 0.9) },
      { label: "S_pol", value: pt.Spol, color: C.ink, a: ap(T, cSum, 0.4), grow: ap(T, cSum + 0.2, 0.9) },
    ], { max: 0.5, w: 780, rowH: 56 });
  } });
})();

/* 6 · per person */
(function () {
  const st = pstart(6) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "6 · Per-person scores · Eq. 21–22", "Separate how many from how strongly", st);
  const cLarge = cue("simply by being large"), cDiv = cue("Dividing each score"), cAvg = cue("the average weight one member"),
    cRaw = cue("the raw scores differ"), cPer = cue("yet per person"), cId = cue("almost identical");
  const waf = svgBox(root, 880, 150, 330, 330);
  const wl = box(root, "", 1250, 190, 560, `
    <div class="legend-row"><i style="background:${C["camp-x"]}"></i>AKP-leaning <span>π<sub>X</sub> = 35.2%</span></div>
    <div class="legend-row" style="margin-top:22px"><i style="background:${C.diag}"></i>same rating <span>20.2%</span></div>
    <div class="legend-row" style="margin-top:22px"><i style="background:${C["camp-y"]}"></i>CHP-leaning <span>π<sub>Y</sub> = 44.6%</span></div>`);
  const rawT = box(root, "label", 880, 540, 440, "Raw score S");
  const raw = svgBox(root, 880, 584, 440, 130);
  const perT = box(root, "label", 1360, 540, 450, "Per person S* × 10³ · 95% CI");
  const per = svgBox(root, 1360, 584, 450, 160);
  const e1 = eq(root, 110, 392, String.raw`S^*_X=\frac{S_X}{\pi_X}\qquad S^*_Y=\frac{S_Y}{\pi_Y}`, 46);
  const e2 = eq(root, 110, 502, String.raw`S^*_X=\frac{1}{100}\cdot\frac{\sum_{i>j}w_{ij}\,a_{ij}}{n_X}`, 46);
  const n2 = box(root, "note", 110, 616, 760, "the average weight carried by one member of the camp");
  const n3 = box(root, "note", 110, 720, 760, `<b>Δ* = S*<sub>X</sub> − S*<sub>Y</sub> = 0.02 × 10⁻³</b>, 95% CI [−0.29, 0.35] × 10⁻³. The raw gap is camp size, not intensity.`);
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    fade(e1, ap(T, cDiv, 0.6)); fade(e2, ap(T, cAvg - 0.3, 0.6)); fade(n2, ap(T, cAvg, 0.6)); fade(n3, ap(T, cId, 0.6));
    fade(wl, ap(T, cLarge + 0.6, 0.6));
    waf.replaceChildren();
    const colOf = k => (k < 35 ? C["camp-x"] : k < 55 ? C.diag : C["camp-y"]);
    for (let k = 0; k < 100; k++) {
      const a = ap(T, cLarge - 0.2 + k * 0.012, 0.35);
      if (a <= 0) continue;
      S("circle", { cx: 16 + (k % 10) * 33, cy: 16 + Math.floor(k / 10) * 33, r: 12.5 * a, fill: colOf(k) }, waf);
    }
    fade(rawT, ap(T, cRaw, 0.5)); fade(raw.parentElement, ap(T, cRaw, 0.5));
    fade(perT, ap(T, cPer, 0.5)); fade(per.parentElement, ap(T, cPer, 0.5));
    hbars(raw, [
      { label: "S_X", value: 0.200, color: C["camp-x"], a: 1, grow: ap(T, cRaw + 0.1, 0.8) },
      { label: "S_Y", value: 0.253, color: C["camp-y"], a: 1, grow: ap(T, cRaw + 0.3, 0.8) },
    ], { max: 0.3, w: 440, labelW: 70, rowH: 62 });
    dotci(per, [
      { label: "S*_X", value: 5.68, lo: 5.43, hi: 5.95, ci: ap(T, cPer + 0.5, 0.6), color: C["camp-x"], a: ap(T, cPer + 0.1, 0.5) },
      { label: "S*_Y", value: 5.66, lo: 5.44, hi: 5.88, ci: ap(T, cPer + 0.7, 0.6), color: C["camp-y"], a: ap(T, cPer + 0.3, 0.5) },
    ], { d0: 5.2, d1: 6.2, w: 450, labelW: 80, rowH: 62 });
  } });
})();

/* 7 · consensus */
(function () {
  const st = pstart(7) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "7 · Consensus · Eq. 23–25", "The other diagonal measures consensus", st);
  const cSwap = cue("Swap the main diagonal"), cCons = cue("measures consensus"), cWarm = cue("shared warmth"), cRej = cue("shared rejection");
  const Wc = WC10;
  const mx = matrixHost(root, 1090, 120, { n: 11, cell: 60, left: 40, top: 40, gap: 2, fs: 14 });
  const gl = S("g", {}, mx.svg);
  const L = mx.L, Tp = mx.T, span = 11 * mx.c;
  const lnDiag = S("line", { x1: L, y1: Tp, x2: L + span, y2: Tp + span, class: "guide" }, gl);
  const lnAnti = S("line", { x1: L + span, y1: Tp, x2: L, y2: Tp + span, class: "guide" }, gl);
  mx.regions([{ cells: (i, j) => i + j > 10, color: C["pos-3"] }, { cells: (i, j) => i + j < 10, color: C["neg-3"] }]);
  const [rWarm, rRej] = mx.svg.querySelectorAll(".region");
  const e1 = eq(root, 110, 392, String.raw`d^{\text{con}}_{ij}=\frac{i+j}{N}-1\qquad a^{\text{con}}_{ij}=\frac{N-|i-j|}{N}`, 42);
  const e2 = eq(root, 110, 500, String.raw`S_{\text{con}}=\underbrace{\sum_{i+j<N}\mathrm{WP}_{ij}}_{S_{\text{neg}}}+\underbrace{\sum_{i+j>N}\mathrm{WP}_{ij}}_{S_{\text{pos}}}`, 40);
  const l1 = box(root, "legend-row", 110, 660, 800, `<i style="background:${C["pos-2"]}"></i>both high <span>shared warmth, weight &gt; 0</span>`);
  const l2 = box(root, "legend-row", 110, 726, 800, `<i style="background:${C["neg-2"]}"></i>both low <span>shared rejection, weight &lt; 0</span>`);
  const n1 = box(root, "note", 110, 800, 780, "AKP × CHP: S<sub>con</sub> = −0.199. In the paper, it is negative for all 78 party pairs.");
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    fade(mx.host, ap(T, st + 0.2, 0.6));
    const e = ap(T, cSwap + 0.4, 1.6);
    const V = W10.map((r, i) => r.map((v, j) => lerp(v, Wc[i][j], e)));
    mx.set(V, { fill: v => div(v), text: v => (Math.abs(v) < 0.005 ? "0" : fc(v, 2)) });
    lnDiag.style.opacity = (1 - e) * 0.8; lnAnti.style.opacity = e * 0.8;
    rWarm.style.opacity = ap(T, cWarm, 0.5); rRej.style.opacity = ap(T, cRej, 0.5);
    fade(e1, ap(T, cSwap + 0.3, 0.6)); fade(e2, ap(T, cCons, 0.6));
    fade(l1, ap(T, cWarm, 0.5)); fade(l2, ap(T, cRej, 0.5)); fade(n1, ap(T, cRej + 1.2, 0.6));
  } });
})();

/* 8 · bootstrap */
(function () {
  const st = pstart(8) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "8 · Bootstrap · Eq. 26–28", "Resample to measure uncertainty", st);
  const cRes = cue("Resample whole respondents"), cRun = cue("rerun everything"), cSE = cue("read standard errors"), cCI = cue("percentile intervals");
  // 2,000 replicates of S_pol, resampling whole respondents (seed 42)
  const n = total(SIM), cellOf = new Int16Array(n);
  { let k = 0; SIM.forEach((r, i) => r.forEach((c, j) => { for (let q = 0; q < c; q++) cellOf[k++] = i * 11 + j; })); }
  const Wf = W10.flat(), R = rng(42), REPS = [];
  for (let b = 0; b < 2000; b++) {
    const cnt = new Int32Array(121);
    for (let k = 0; k < n; k++) cnt[cellOf[Math.floor(R() * n)]]++;
    let s = 0; for (let c = 0; c < 121; c++) s += Wf[c] * cnt[c];
    REPS.push(s / n);
  }
  const sorted = Float64Array.from(REPS).sort();
  const mean = REPS.reduce((a, b) => a + b, 0) / REPS.length, se = sd(REPS), lo = quantile(sorted, 0.025), hi = quantile(sorted, 0.975);
  const point = polScores(SIM, W10).Spol;
  const d0 = 0.42, d1 = 0.488, bw = 0.0025, nb = Math.round((d1 - d0) / bw);
  const finalCounts = new Array(nb).fill(0); REPS.forEach(v => { const k = Math.floor((v - d0) / bw); if (k >= 0 && k < nb) finalCounts[k]++; });
  const ymax = Math.max(...finalCounts) * 1.12;
  const hs = svgBox(root, 880, 150, 930, 640);
  const X = axisLinear(d0, d1, 70, 910), Y = axisLinear(0, ymax, 560, 110);
  const e1 = eq(root, 110, 392, String.raw`k^*_1,\dots,k^*_n\ \overset{\text{iid}}{\sim}\ \text{Uniform}\{1,\dots,n\}`, 40);
  const e2 = eq(root, 110, 486, String.raw`\widehat{\mathrm{SE}}=\mathrm{SD}\bigl(S^{(1)},\dots,S^{(B)}\bigr)`, 42);
  const e3 = eq(root, 110, 580, String.raw`\mathrm{CI}_{95\%}=\bigl[\hat q_{0.025},\ \hat q_{0.975}\bigr]`, 42);
  const n1 = box(root, "note", 110, 684, 780, "Whole respondents are redrawn, so each pair of ratings stays together. W never changes; only P does.");
  const n2 = box(root, "note", 110, 800, 780, `Published for AKP × CHP: <b>S<sub>pol</sub> = 0.453</b>, SE 0.008, 95% CI [0.436, 0.469].`);
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    fade(e1, ap(T, cRes, 0.6)); fade(n1, ap(T, cRes + 1.2, 0.6)); fade(e2, ap(T, cSE, 0.6)); fade(e3, ap(T, cCI, 0.6)); fade(n2, ap(T, cCI + 1.4, 0.6));
    fade(hs.parentElement, ap(T, cRes - 0.3, 0.6));
    const B = Math.round(2000 * Math.pow(clamp((T - cRes) / (cRun + 1.6 - cRes), 0, 1), 1.9));
    const counts = new Array(nb).fill(0);
    for (let b = 0; b < B; b++) { const k = Math.floor((REPS[b] - d0) / bw); if (k >= 0 && k < nb) counts[k]++; }
    hs.replaceChildren();
    const ticks = [0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48];
    ticks.forEach(v => S("text", { x: X(v), y: 590, class: "tick", "text-anchor": "middle", "font-size": 19 }, hs).textContent = f(v, 2));
    S("line", { x1: 70, x2: 910, y1: 560, y2: 560, stroke: C.rule, "stroke-width": 2 }, hs);
    S("text", { x: 490, y: 630, class: "axt", "text-anchor": "middle", "font-size": 21 }, hs).textContent = "S_pol in each bootstrap replicate";
    counts.forEach((c, k) => { if (c) S("rect", { x: X(d0 + k * bw) + 1.5, y: Y(c), width: X(d0 + bw) - X(d0) - 3, height: 560 - Y(c), rx: 2, fill: C["ink-2"], opacity: 0.8 }, hs); });
    S("line", { x1: X(point), x2: X(point), y1: 96, y2: 560, stroke: C.pen, "stroke-width": 2.5, "stroke-dasharray": "7 5" }, hs);
    const cnt = S("text", { x: 910, y: 40, "text-anchor": "end", "font-size": 30 }, hs);
    cnt.textContent = `B = ${B.toLocaleString("en")} / 2,000`;
    const aSE = ap(T, cSE, 0.6), aCI = ap(T, cCI, 0.6);
    if (aSE > 0) {
      const g = S("g", { opacity: aSE * (1 - 0.6 * aCI) }, hs);
      S("line", { x1: X(mean - se), x2: X(mean + se), y1: 80, y2: 80, stroke: C.ink, "stroke-width": 6 }, g);
      S("text", { x: X(mean + se) + 12, y: 80, "dominant-baseline": "central", "font-size": 22, fill: C["ink-2"] }, g).textContent = `±1 SE = ${f(se, 4)}`;
    }
    if (aCI > 0) {
      const g = S("g", { opacity: aCI }, hs);
      S("rect", { x: X(lo), y: 96, width: X(hi) - X(lo), height: 464, fill: C.pen, opacity: 0.07 }, g);
      [lo, hi].forEach(v => S("line", { x1: X(v), x2: X(v), y1: 96, y2: 560, stroke: C.pen, "stroke-width": 2.5 }, g));
      S("text", { x: X(lo) - 10, y: 116, "text-anchor": "end", "font-size": 22, fill: C.pen }, g).textContent = f(lo, 3);
      S("text", { x: X(hi) + 10, y: 116, "font-size": 22, fill: C.pen }, g).textContent = f(hi, 3);
    }
  } });
})();

/* 9 · summary */
(function () {
  const st = pstart(9) - 0.45;
  const root = box(stage, "scene", 0, 0, 1920);
  const hd = header(root, "Summary", "One table, one weight matrix", st);
  const cT = cue("One table"), cW = cue("one weight matrix"), cS = cue("and scores that separate"), cHow = cue("how many people disagree"), cPkg = cue("available as the bipolarization package");
  const steps = ["Tabulate the pair of ratings", "Weight every cell by distance × agreement", "Normalise the whole table once", "Sum each triangle: S<sub>X</sub>, S<sub>Y</sub>, S<sub>pol</sub>", "Divide by camp size: S*", "Bootstrap whole respondents"];
  const list = steps.map((s, k) => box(root, "legend-row", 110, 390 + k * 66, 780, `<b class="mono" style="color:var(--pen);font-size:26px;width:30px">${k + 1}</b><span style="color:var(--ink)">${s}</span>`));
  const mA = matrixHost(root, 880, 210, { n: 11, cell: 22, left: 4, top: 4, gap: 1.2, ticks: false });
  const mW = matrixHost(root, 1210, 210, { n: 11, cell: 22, left: 4, top: 4, gap: 1.2, ticks: false });
  const bars = svgBox(root, 1540, 230, 270, 210);
  const ar = [box(root, "op", 1136, 270, null, "→"), box(root, "op", 1466, 270, null, "→")];
  const caps = [box(root, "cap", 880, 470, 250, "joint table A"), box(root, "cap", 1210, 470, 250, "weights W"), box(root, "cap", 1540, 470, 270, "scores S and S*")];
  const how1 = box(root, "legend-row", 880, 560, 930, `<i style="background:${C.ink}"></i>how many disagree <span>→ S<sub>X</sub>, S<sub>Y</sub>, S<sub>pol</sub></span>`);
  const how2 = box(root, "legend-row", 880, 626, 930, `<i style="background:${C.pen}"></i>how strongly <span>→ S*<sub>X</sub>, S*<sub>Y</sub></span>`);
  const chip = box(root, "chip", 880, 724, null, "pip install bipolarization");
  const rnote = box(root, "note", 880, 812, 900, "Python package <b>bipolarization</b> · R port <b>bipolarizationR</b>");
  mA.set(SIM, { fill: v => (v ? ramp("mass", Math.sqrt(v / 311)) : C.sheet) });
  mW.set(W10, { fill: v => ramp("pos", v) });
  const pt = polScores(SIM, W10);
  SCENES.push({ root, start: st, update(T) {
    hd(T);
    list.forEach((e, k) => fade(e, ap(T, cT + 0.2 + k * 0.35, 0.5)));
    fade(mA.host, ap(T, cT, 0.5)); fade(caps[0], ap(T, cT, 0.5));
    fade(ar[0], ap(T, cW - 0.1, 0.4)); fade(mW.host, ap(T, cW, 0.5)); fade(caps[1], ap(T, cW, 0.5));
    fade(ar[1], ap(T, cS - 0.1, 0.4)); fade(bars.parentElement, ap(T, cS, 0.5)); fade(caps[2], ap(T, cS, 0.5));
    fade(how1, ap(T, cHow, 0.5)); fade(how2, ap(T, cHow + 0.9, 0.5));
    fade(chip, ap(T, cPkg, 0.6)); fade(rnote, ap(T, cPkg + 0.6, 0.6));
    hbars(bars, [
      { label: "S_X", value: pt.SX, color: C["camp-x"], a: 1, grow: ap(T, cS + 0.1, 0.7) },
      { label: "S_Y", value: pt.SY, color: C["camp-y"], a: 1, grow: ap(T, cS + 0.25, 0.7) },
      { label: "S_pol", value: pt.Spol, color: C.ink, a: 1, grow: ap(T, cS + 0.4, 0.7) },
    ], { max: 0.5, w: 270, labelW: 70, rowH: 64, fmt: v => f(v, 2) });
  } });
})();

/* ---------- chrome: brand, subtitles, progress, end card ---------- */
const brand = box(stage, "brand", 110, 50, null, "bi<b>polar</b>ization<span>method explainer</span>");
brand.style.right = "auto";
const sub = box(stage, "", 0, 0, null, "");
sub.id = "sub"; sub.style.left = ""; sub.style.top = "";
const prog = box(stage, "", 0, 0, null, "");
prog.id = "prog"; prog.style.left = ""; prog.style.top = "";
const progBars = SCENES.slice(1).map(() => { const i = document.createElement("i"); const b = document.createElement("b"); i.appendChild(b); prog.appendChild(i); return b; });
const end = box(stage, "", 0, 0, null, `<div class="eyebrow">bipolarization · Python and R</div><div class="big-title">Two camps, one table</div>
  <div class="credits">Voice-over generated with ElevenLabs. Results from the paper's application to the 2024 Turkey General Social Survey;<br>the AKP × CHP cell counts shown are simulated to reproduce the published figures.</div>`);
end.id = "end"; end.style.left = ""; end.style.top = "";
const NARR_END = LEAD + AUDIO_DUR;

const CHUNKS = (() => {
  const fix = s => s.replace("eleven by eleven", "eleven-by-eleven").replace("five point", "five-point").replace("per person score", "per-person score")
    .replace("anti diagonal", "anti-diagonal").replace("distributions and still", "distributions… and still").replace("differ yet", "differ… yet");
  const txt = ts => ts.map(t => t.t).join(" ");
  function split(ts) {
    const text = txt(ts);
    if (text.length <= 74) return [ts];
    let best = null;
    for (let k = 1; k < ts.length; k++) {
      const left = txt(ts.slice(0, k)).length, right = text.length - left - 1;
      let score = Math.abs(left - right);
      if (/[,;:]$/.test(ts[k - 1].t)) score -= 22;
      if (left < 16 || right < 16) score += 200;
      if (!best || score < best.score) best = { k, score };
    }
    return [...split(ts.slice(0, best.k)), ...split(ts.slice(best.k))];
  }
  const sentences = [];
  let cur = [];
  TOK.forEach(t => {
    if (cur.length && t.para !== cur[0].para) { sentences.push(cur); cur = []; }
    cur.push(t);
    if (/[.?!]$/.test(t.t)) { sentences.push(cur); cur = []; }
  });
  if (cur.length) sentences.push(cur);
  const out = [];
  sentences.forEach(sn => split(sn).forEach(ts => out.push({ s: ts[0].s + LEAD, e: ts[ts.length - 1].e + LEAD, text: fix(txt(ts)) })));
  out.forEach((c, i) => { const nx = out[i + 1]; c.show = c.s - 0.1; c.hide = nx && nx.s - c.e < 1.2 ? nx.s - 0.1 : c.e + 0.5; });
  return out;
})();

function renderAt(T) {
  SCENES.forEach((sc, i) => {
    const nx = SCENES[i + 1];
    const a = (i === 0 ? ap(T, 0, 0.5) : ap(T, sc.start, 0.5)) * (nx ? 1 - ap(T, nx.start - 0.35, 0.35) : 1);
    sc.root.style.opacity = a.toFixed(3);
    sc.root.style.visibility = a > 0.002 ? "visible" : "hidden";
    if (a > 0.002) sc.update(T);
  });
  const ch = CHUNKS.find(c => T >= c.show && T < c.hide);
  sub.textContent = ch ? ch.text : "";
  sub.style.opacity = ch ? "1" : "0";
  SCENES.slice(1).forEach((sc, k) => {
    const nx = SCENES[k + 2], s = sc.start, e = nx ? nx.start : NARR_END;
    progBars[k].style.width = (100 * clamp((T - s) / (e - s), 0, 1)).toFixed(2) + "%";
  });
  const ea = ap(T, NARR_END + 0.25, 0.9);
  end.style.opacity = ea.toFixed(3);
  end.style.visibility = ea > 0 ? "visible" : "hidden";
  brand.style.opacity = (1 - ea).toFixed(3);
  prog.style.opacity = (1 - ea).toFixed(3);
}

async function boot() {
  readPalette();
  await Promise.all(["800 80px Archivo", "600 24px Archivo", "400 30px 'Source Serif 4'", "500 30px 'Source Serif 4'", "500 20px 'IBM Plex Mono'"].map(s => document.fonts.load(s)));
  await document.fonts.ready;
  const t0 = Date.now();
  while (!window.__mjReady && Date.now() - t0 < 30000) await new Promise(r => setTimeout(r, 100));
  EQS.forEach(e => { e.replaceChildren(MathJax.tex2svg(e.dataset.tex, { display: false })); });
  document.head.appendChild(MathJax.svgStylesheet());
  renderAt(0);
  window.__video = { FPS, TOTAL, frames: Math.ceil(TOTAL * FPS), chunks: CHUNKS, cues: SCENES.map(s => s.start) };
  window.__videoReady = true;
}
window.renderAt = renderAt;
boot();
