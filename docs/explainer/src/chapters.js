/* ============================================================
   Chapters
   ============================================================ */
const WP4 = weights(4).W;                                   // toy polarization weights
const WC4 = weights(4, { type: "consensus" }).W;            // toy consensus weights
const W10 = weights(10).W, WC10 = weights(10, { type: "consensus" }).W;
const TOY_LIST = (() => {                                   // the 20 toy respondents, in a fixed shuffled order
  const out = [];
  TOY.forEach((r, i) => r.forEach((c, j) => { for (let k = 0; k < c; k++) out.push({ i, j }); }));
  const R = rng(3);
  for (let k = out.length - 1; k > 0; k--) { const m = Math.floor(R() * (k + 1)); [out[k], out[m]] = [out[m], out[k]]; }
  return out.map((o, k) => ({ ...o, id: "R" + String(k + 1).padStart(2, "0") }));
})();
const campOf = (i, j) => (i > j ? "x" : i < j ? "y" : "d");
const SUB = s => `S<sub>${s}</sub>`;
/* "S*_X" -> S* with a subscript X, inside an SVG <text> */
function svgSub(el, label) {
  el.replaceChildren();
  const m = String(label).match(/^(.*?)_(\w+)(.*)$/);
  if (!m) { el.textContent = label; return; }
  el.appendChild(document.createTextNode(m[1]));
  const t = S("tspan", { "baseline-shift": "sub", "font-size": "0.75em" }, el); t.textContent = m[2];
  if (m[3]) el.appendChild(document.createTextNode(m[3]));
}

function RO(host, items) {
  host.innerHTML = items.map(it =>
    `<div class="ro${it.hl ? " hl" : ""}"><div class="k">${it.sw ? `<i class="sw ${it.sw}"></i>` : ""}<span>${it.k}</span></div><div class="v">${it.v}</div>${it.s ? `<div class="s">${it.s}</div>` : ""}</div>`
  ).join("");
}
function tokens(host, list) {
  host.innerHTML = "";
  return list.map(r => {
    const e = H("span", { class: "tok" }, `<i>${r.id}</i>${r.i}·${r.j}<span class="cnt"></span>`);
    host.appendChild(e);
    return e;
  });
}
function cellTip(i, j, lines, ev) { showTip(`<b>(${i}, ${j})</b><br>${lines.join("<br>")}`, ev); }

/* signed horizontal bars with a zero line */
function Bars(host, { lo, hi, width = 380, row = 24, labelW = 64, fmt = v => f(v, 4) }) {
  const svg = S("svg", { class: "chart", role: "img" });
  host.appendChild(svg);
  return (rows) => {
    svg.replaceChildren();
    const Ht = rows.length * row + 8;
    svg.setAttribute("viewBox", `0 0 ${width} ${Ht}`);
    const x = axisLinear(lo, hi, labelW, width - 64);
    S("line", { x1: x(0), x2: x(0), y1: 0, y2: Ht, class: "axis" }, svg);
    rows.forEach((r, k) => {
      const y = k * row + 4;
      const lab = S("text", { x: labelW - 8, y: y + row / 2, "text-anchor": "end", "dominant-baseline": "central", class: "axt" }, svg);
      svgSub(lab, r.label);
      const v = clamp(r.value, lo, hi), x0 = Math.min(x(0), x(v)), x1 = Math.max(x(0), x(v));
      S("rect", { x: x0, y: y + 3, width: Math.max(0.5, x1 - x0), height: row - 8, rx: 2, fill: r.color, style: "transition: x .4s, width .4s" }, svg);
      const t = S("text", { x: v >= 0 ? x1 + 6 : x(0) + 6, y: y + row / 2, "dominant-baseline": "central", "text-anchor": "start", "font-size": 11.5 }, svg);
      t.textContent = r.text ?? fmt(r.value);
    });
  };
}

/* ---------------- 1. The problem ---------------- */
function chProblem() {
  const m = [0.37, 0.03, 0.03, 0.04, 0.04, 0.10, 0.05, 0.06, 0.07, 0.04, 0.17];
  const K = 11, F = [];
  m.reduce((s, v, k) => (F[k] = s + v), 0);
  const Fm = k => (k < 0 ? 0 : F[k]);
  const ind = zeros(K), co = zeros(K), ctr = zeros(K);
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    ind[i][j] = m[i] * m[j];
    co[i][j] = i === j ? m[i] : 0;
    ctr[i][j] = Math.max(0, Math.min(Fm(i), 1 - Fm(j - 1)) - Math.max(Fm(i - 1), 1 - Fm(j)));
  }
  const joint = t => ind.map((r, i) => r.map((v, j) => (t >= 0 ? (1 - t) * v + t * co[i][j] : (1 + t) * v - t * ctr[i][j])));
  const mean = m.reduce((s, v, k) => s + v * k, 0), sdv = Math.sqrt(m.reduce((s, v, k) => s + v * (k - mean) ** 2, 0));
  const host = $("#probFig");
  const mx = MatrixView(host, { n: K, cell: 26, left: 78, top: 74, gap: 1, ticks: true, rowTitle: () => T("X rating (i)", "X puanı (i)"), colTitle: () => T("Y rating (j)", "Y puanı (j)"), rowTitleX: 10, colTitleY: 10,
    onHover: (i, j, ev) => cellTip(i, j, [`p = ${f(100 * cur[i][j], 1)}%`], ev), onMove: (i, j, ev) => cellTip(i, j, [`p = ${f(100 * cur[i][j], 1)}%`], ev) });
  const gM = S("g", {}, mx.svg);
  const L = mx.L, Tp = mx.T, c = mx.c;
  function margins() {
    gM.replaceChildren();
    m.forEach((v, k) => {
      const len = v / 0.37 * 44;
      S("rect", { x: L - 22 - len, y: Tp + k * c + 4, width: len, height: c - 8, rx: 1.5, fill: C.diag }, gM);
      S("rect", { x: L + k * c + 4, y: Tp - 18 - len, width: c - 8, height: len, rx: 1.5, fill: C.diag }, gM);
    });
  }
  let cur = joint(0), t = 0;
  const out = $("#couplingOut"), slider = $("#coupling");
  function render() {
    cur = joint(t);
    mx.set(cur, { fill: v => ramp("mass", Math.sqrt(v / 0.37)) });
    margins();
    out.textContent = f(t, 2);
    const sp = polScores(cur, W10), sc = conScores(cur, WC10);
    RO($("#probRO"), [
      { k: T("X: mean · SD", "X: ort. · SS"), v: `${f(mean, 2)} · ${f(sdv, 2)}` },
      { k: T("Y: mean · SD", "Y: ort. · SS"), v: `${f(mean, 2)} · ${f(sdv, 2)}` },
      { k: SUB("pol"), v: f(sp.Spol, 3), hl: true },
      { k: SUB("con"), v: f(sc.Scon, 3) },
    ]);
  }
  function setT(v) { t = clamp(v, -1, 1); slider.value = t; render(); }
  slider.addEventListener("input", () => { ch.stop(); setT(+slider.value); });
  async function animateTo(v, ms, ctx) {
    const a = t; mx.live(true);
    await tween(ms, e => { if (ctx.alive(ctx.token)) setT(lerp(a, v, e)); });
    mx.live(false);
  }
  const ch = Chapter("problem", [
    { enter: ctx => { mx.guide(null); animateTo(0, ctx.instant ? 0 : 700, ctx); } },
    { enter: ctx => { mx.guide("diag"); animateTo(1, 1400, ctx); }, dur: 3600 },
    { enter: ctx => { mx.guide("anti"); animateTo(-1, 1800, ctx); }, dur: 4400 },
    { enter: ctx => { mx.guide(null); animateTo(-0.6, 900, ctx); } },
  ]);
  onTheme(render); onLang(render);
  render();
  ch.go(0, { instant: true });
}

/* ---------------- 2. Joint table ---------------- */
function chTable() {
  const host = $("#tableFig");
  const toks = tokens($("#toyTokens"), TOY_LIST);
  let A = zeros(5);
  const mx = MatrixView(host, { n: 5, cell: 58, left: 36, top: 36, fs: 19, rowTitle: () => T("i · rating of X", "i · X puanı"), colTitle: () => T("j · rating of Y", "j · Y puanı"),
    onHover: (i, j, ev) => cellTip(i, j, [`a = ${A[i][j]}`], ev), onMove: (i, j, ev) => cellTip(i, j, [`a = ${A[i][j]}`], ev) });
  let regionMode = null;
  function render() {
    mx.set(A, { fill: v => ramp("mass", v / 4.5), text: v => (v ? String(v) : "") });
    if (regionMode === "pol") {
      mx.regions([
        { cells: (i, j) => i > j, color: C["camp-x"] },
        { cells: (i, j) => i < j, color: C["camp-y"] },
      ]);
      mx.guide("diag");
      RO($("#tableRO"), [
        { k: T("X-leaning n<sub>X</sub>", "X'e yakın n<sub>X</sub>"), v: "6", sw: "x" },
        { k: T("diagonal", "köşegen"), v: "7", sw: "d" },
        { k: T("Y-leaning n<sub>Y</sub>", "Y'ye yakın n<sub>Y</sub>"), v: "7", sw: "y" },
      ]);
    } else if (regionMode === "con") {
      mx.regions([
        { cells: (i, j) => i + j < 4, color: C["neg-2"] },
        { cells: (i, j) => i + j > 4, color: C["pos-2"] },
      ]);
      mx.guide("anti");
      RO($("#tableRO"), [
        { k: "i + j < N", v: "8", sw: "neg" },
        { k: T("anti-diagonal", "ters köşegen"), v: "11", sw: "d" },
        { k: "i + j > N", v: "1", sw: "pos" },
      ]);
    } else {
      mx.regions([]); mx.guide(null);
      RO($("#tableRO"), [{ k: "s<sub>0</sub>", v: "0" }, { k: "N", v: "4" }, { k: "n", v: String(total(A)) + " / 20", hl: true }]);
    }
    toks.forEach((e, k) => {
      const r = TOY_LIST[k];
      e.className = "tok" + (regionMode === "pol" ? " " + campOf(r.i, r.j) : "");
      if (regionMode === "con") e.className = "tok " + (r.i + r.j < 4 ? "x" : r.i + r.j > 4 ? "y" : "d");
    });
  }
  async function tally(ctx) {
    A = zeros(5); regionMode = null; render();
    toks.forEach(e => e.classList.add("gone"));
    if (REDUCED || ctx.instant) { A = TOY.map(r => r.slice()); toks.forEach(e => e.classList.remove("gone")); render(); return; }
    toks.forEach(e => e.classList.remove("gone"));
    for (let k = 0; k < TOY_LIST.length; k++) {
      if (!ctx.alive(ctx.token)) return;
      const r = TOY_LIST[k], e = toks[k], a = e.getBoundingClientRect(), b = mx.cells[r.i][r.j].r.getBoundingClientRect();
      const fly = e.cloneNode(true);
      Object.assign(fly.style, { position: "fixed", left: a.left + "px", top: a.top + "px", margin: 0, zIndex: 30, pointerEvents: "none", background: C.sheet });
      document.body.appendChild(fly);
      e.classList.add("gone");
      const dx = b.left + b.width / 2 - (a.left + a.width / 2), dy = b.top + b.height / 2 - (a.top + a.height / 2);
      fly.animate([{ transform: "translate(0,0)", opacity: 1 }, { transform: `translate(${dx}px,${dy}px) scale(.5)`, opacity: 0.3 }], { duration: 520, easing: "cubic-bezier(.3,.7,.3,1)" })
        .finished.then(() => { fly.remove(); if (!ctx.alive(ctx.token)) return; A[r.i][r.j]++; e.classList.remove("gone"); render(); });
      await sleep(110);
    }
  }
  Chapter("table", [
    { enter: ctx => { A = zeros(5); regionMode = null; toks.forEach(e => e.classList.remove("gone")); render(); } },
    { enter: ctx => tally(ctx), dur: 4600 },
    { enter: () => { A = TOY.map(r => r.slice()); regionMode = "pol"; render(); } },
    { enter: () => { A = TOY.map(r => r.slice()); regionMode = "con"; render(); } },
  ]).go(0, { instant: true });
  onTheme(render); onLang(render);
}

/* ---------------- 3. Weights ---------------- */
function chWeights() {
  let geom = "pol", N = 4, sel = [1, 3], focus = 3, trio = [];
  const host = $("#wTrio");
  function terms(i, j) {
    if (geom === "pol") return { d: Math.abs(i - j) / N, a: 1 - Math.abs(i + j - N) / N };
    return { d: (i + j) / N - 1, a: (N - Math.abs(i - j)) / N };
  }
  function build() {
    host.replaceChildren(); trio = [];
    const caps = N === 4 ? ["d × 4", "a × 4", "m × 16"] : ["d", "a", "m = d · a"];
    caps.forEach((cap, k) => {
      const wrap = H("div", { class: "blk" });
      wrap.appendChild(H("div", { class: "cap" }, cap));
      const box = H("div"); wrap.appendChild(box);
      host.appendChild(wrap);
      if (k < 2) host.appendChild(H("div", { class: "op" }, k === 0 ? "×" : "="));
      const v = MatrixView(box, { n: N + 1, cell: N === 4 ? 34 : 16, left: N === 4 ? 16 : 16, top: N === 4 ? 16 : 16, fs: 13, gap: N === 4 ? 1.5 : 1,
        ticks: N === 4,
        onHover: (i, j) => { sel = [i, j]; inspect(); ping(); },
        onClick: (i, j) => { sel = [i, j]; inspect(); ping(); } });
      trio.push({ wrap, v });
    });
  }
  function ping() {
    const list = focus === 2 && N === 10 ? [[0, 5], [3, 8], [5, 10]] : [sel];
    trio.forEach(t => t.v.ping(list));
  }
  function render() {
    const n = N + 1;
    const D = [], Aa = [], M = [];
    for (let i = 0; i < n; i++) { D.push([]); Aa.push([]); M.push([]); for (let j = 0; j < n; j++) { const t = terms(i, j); D[i].push(t.d); Aa[i].push(t.a); M[i].push(t.d * t.a); } }
    const dFill = v => (geom === "pol" ? ramp("pos", v) : div(v));
    const txt = scale => (N === 4 ? v => String(Math.round(v * scale)).replace("-", MINUS) : null);
    trio[0].v.set(D, { fill: dFill, text: txt(4) });
    trio[1].v.set(Aa, { fill: v => ramp("mass", v * 0.8), text: txt(4) });
    trio[2].v.set(M, { fill: dFill, text: txt(16) });
    trio.forEach((t, k) => { t.wrap.style.opacity = focus === 0 && k > 0 ? 0.3 : focus === 1 && k === 2 ? 0.3 : 1; t.wrap.style.transition = "opacity .3s"; });
    trio.forEach(t => t.v.guide(geom === "pol" ? "anti" : "diag"));
    $("#wCap").innerHTML = geom === "pol"
      ? T(`Colour: <i class="sw pos"></i> 0 → 1. Dashed line: the anti-diagonal i + j = N, where agreement is 1. ${N === 4 ? "Numbers are whole multiples: d and a in quarters, m in sixteenths." : "Hover a cell to see its terms."}`,
          `Renk: <i class="sw pos"></i> 0 → 1. Kesikli çizgi: uyumun 1 olduğu ters köşegen i + j = N. ${N === 4 ? "Sayılar tam katlardır: d ve a dörtte bir, m on altıda bir birimindedir." : "Terimleri görmek için bir hücrenin üzerine gelin."}`)
      : T(`Colour: <i class="sw neg"></i> −1 → 0 → +1 <i class="sw pos"></i>. Dashed line: the main diagonal, where agreement is 1.`,
          `Renk: <i class="sw neg"></i> −1 → 0 → +1 <i class="sw pos"></i>. Kesikli çizgi: uyumun 1 olduğu ana köşegen.`);
    inspect(); ping();
  }
  function inspect() {
    const box = $("#wInspect");
    if (focus === 2 && N === 10 && geom === "pol") {
      const rows = [[0, 5], [3, 8], [5, 10]].map(([i, j]) => { const t = terms(i, j); return `(${i},${j})&:\\ d=${f(t.d, 2)},\\ a=${f(t.a, 2)},\\ m=${f(t.d * t.a, 2)}`; });
      box.innerHTML = `<div class="hd">${T("Five points apart, three different weights", "Beş puan aralık, üç farklı ağırlık")}</div><div class="m"></div>`;
      tex(box.querySelector(".m"), `\\begin{aligned}${rows.join("\\\\")}\\end{aligned}`, true);
      return;
    }
    const [i, j] = sel, t = terms(i, j);
    const ex = [];
    if (geom === "pol") {
      ex.push(`d_{${i},${j}}=\\frac{|${i}-${j}|}{${N}}=${trim(t.d)}`);
      if (focus >= 1) ex.push(`a_{${i},${j}}=1-\\frac{|${i}+${j}-${N}|}{${N}}=${trim(t.a)}`);
    } else {
      ex.push(`d_{${i},${j}}=\\frac{${i}+${j}}{${N}}-1=${trim(t.d)}`);
      ex.push(`a_{${i},${j}}=\\frac{${N}-|${i}-${j}|}{${N}}=${trim(t.a)}`);
    }
    if (focus >= 2 || geom === "con") ex.push(`m_{${i},${j}}=${trim(t.d)}\\times${trim(t.a)}=${trim(t.d * t.a)}`);
    box.innerHTML = `<div class="hd">${T("Selected cell — hover or tap any cell", "Seçili hücre — herhangi bir hücrenin üzerine gelin ya da dokunun")}</div><div class="m"></div>`;
    tex(box.querySelector(".m"), ex.join("\\qquad "), false);
  }
  const trim = v => { const s = f(v, 4); return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s; };
  function setSeg(id, v) { $$(`#${id} button`).forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(v)))); }
  function setState(g, n) {
    const rebuild = n !== N;
    geom = g; N = n; setSeg("wGeom", g); setSeg("wN", n);
    if (sel[0] > N || sel[1] > N) sel = [1, 3];
    if (rebuild) build();
    render();
  }
  $$("#wGeom button").forEach(b => b.addEventListener("click", () => { ch.stop(); setState(b.dataset.v, N); }));
  $$("#wN button").forEach(b => b.addEventListener("click", () => { ch.stop(); setState(geom, +b.dataset.v); }));
  build();
  const ch = Chapter("weights", [
    { enter: () => { focus = 0; sel = [0, 4]; setState("pol", 4); } },
    { enter: () => { focus = 1; sel = [1, 3]; setState("pol", 4); } },
    { enter: () => { focus = 2; sel = [3, 8]; setState("pol", 10); }, dur: 5200 },
    { enter: () => { focus = 3; sel = [0, 0]; setState("con", 4); } },
  ]);
  ch.go(0, { instant: true });
  onTheme(render); onLang(render);
}

/* ---------------- 4. Kernels ---------------- */
function chKernels() {
  const st = { kernel: "power", p: 1, sigma: 0.5, q: 1 };
  const mxHost = $("#kMatrix");
  let Wcur = W10, Mcur = W10;
  const mx = MatrixView(mxHost, { n: 11, cell: 22, left: 22, top: 22, gap: 1, fs: 8,
    onHover: (i, j, ev) => cellTip(i, j, [`m = ${f(Mcur[i][j], 4)}`, `w = ${f(Wcur[i][j], 4)}`], ev),
    onMove: (i, j, ev) => cellTip(i, j, [`m = ${f(Mcur[i][j], 4)}`, `w = ${f(Wcur[i][j], 4)}`], ev) });
  const cv = S("svg", { viewBox: "0 0 320 252", class: "chart", role: "img", "aria-label": "kernel curve" });
  $("#kCurve").appendChild(cv);
  const X = axisLinear(0, 1, 40, 306), Y = axisLinear(0, 1, 196, 14);
  const fd = d => (st.kernel === "power" ? Math.pow(d, st.p) : 1 - Math.exp(-(d * d) / (2 * st.sigma * st.sigma)));
  function drawCurve(maxv) {
    cv.replaceChildren();
    [0, 0.25, 0.5, 0.75, 1].forEach(v => {
      S("line", { x1: X(0), x2: X(1), y1: Y(v), y2: Y(v), class: "gridline" }, cv);
      S("text", { x: X(0) - 6, y: Y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, cv).textContent = v;
      S("text", { x: X(v), y: Y(0) + 14, class: "tick", "text-anchor": "middle" }, cv).textContent = v;
    });
    S("text", { x: X(0.5), y: 226, class: "axt", "text-anchor": "middle" }, cv).textContent = T("distance d", "uzaklık d");
    S("text", { x: 10, y: Y(0.5), class: "axt", "text-anchor": "middle", transform: `rotate(-90 10 ${Y(0.5)})` }, cv).textContent = T("weight", "ağırlık");
    let raw = "", norm = "";
    for (let k = 0; k <= 100; k++) {
      const d = k / 100, v = fd(d);
      raw += (k ? "L" : "M") + X(d).toFixed(1) + " " + Y(v).toFixed(1);
      norm += (k ? "L" : "M") + X(d).toFixed(1) + " " + Y(v / maxv).toFixed(1);
    }
    if (st.kernel === "gaussian" && st.sigma <= 1) {
      S("line", { x1: X(st.sigma), x2: X(st.sigma), y1: Y(0), y2: Y(1), stroke: C.muted, "stroke-dasharray": "2 3" }, cv);
      S("text", { x: X(st.sigma) + 4, y: Y(0.96), "font-size": 11, fill: C.muted }, cv).textContent = "σ";
    }
    S("line", { x1: 40, x2: 58, y1: 244, y2: 244, stroke: C["pos-2"], "stroke-width": 2.2 }, cv);
    S("text", { x: 62, y: 244, "font-size": 10.5, fill: C["ink-2"], "dominant-baseline": "central" }, cv).textContent = T("normalised", "normalize");
    if (Math.abs(maxv - 1) > 1e-4) {
      S("path", { d: raw, fill: "none", stroke: C.muted, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, cv);
      S("line", { x1: 150, x2: 168, y1: 244, y2: 244, stroke: C.muted, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, cv);
      S("text", { x: 172, y: 244, "font-size": 10.5, fill: C["ink-2"], "dominant-baseline": "central" }, cv).textContent = T("raw", "ham") + `, max ${f(maxv, 4)}`;
    }
    S("path", { d: norm, fill: "none", stroke: C["pos-2"], "stroke-width": 2.2 }, cv);
    for (let k = 0; k <= 10; k++) S("circle", { cx: X(k / 10), cy: Y(fd(k / 10) / maxv), r: 3, fill: C["pos-2"], stroke: C.sheet, "stroke-width": 1.5 }, cv);
  }
  function render() {
    const opts = { kernel: st.kernel, p: st.p, q: st.q, sigma: st.sigma };
    const res = weights(10, opts);
    Wcur = res.W; Mcur = res.M;
    mx.set(Wcur, { fill: v => ramp("pos", v) });
    mx.guide("anti");
    drawCurve(fd(1));
    $("#kPout").textContent = f(st.p, 2); $("#kSout").textContent = f(st.sigma, 2); $("#kQout").textContent = f(st.q, 2);
    $("#kPctl").hidden = st.kernel !== "power"; $("#kSctl").hidden = st.kernel !== "gaussian";
    $$("#kKernel button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === st.kernel)));
    const sp = polScores(SIM, Wcur).Spol;
    const match = T6.find(r => r[2].kernel === st.kernel && Math.abs((r[2].q) - st.q) < 1e-9 &&
      (st.kernel === "power" ? Math.abs(r[2].p - st.p) < 1e-9 : Math.abs(r[2].sigma - st.sigma) < 1e-9));
    $$("#kPresets .chip").forEach((b, k) => b.setAttribute("aria-pressed", String(T6[k] === match)));
    RO($("#kRO"), [
      { k: T("largest raw |m|", "en büyük ham |m|"), v: f(res.max, 4) },
      { k: T("rescaling", "yeniden ölçek"), v: `÷ ${f(res.max, 4)}` },
      { k: T("S<sub>pol</sub>, simulated AKP×CHP", "S<sub>pol</sub>, simüle AKP×CHP"), v: f(sp, 3), hl: true },
      { k: T("published (Table 6)", "yayımlanan (Tablo 6)"), v: match ? f(match[4], 3) : "–", s: match ? `[${f(match[5], 3)}, ${f(match[6], 3)}]` : T("pick a preset", "bir hazır tanım seçin") },
    ]);
  }
  const P = $("#kP"), Sg = $("#kS"), Q = $("#kQ");
  P.addEventListener("input", () => { ch.stop(); st.p = +P.value; render(); });
  Sg.addEventListener("input", () => { ch.stop(); st.sigma = +Sg.value; render(); });
  Q.addEventListener("input", () => { ch.stop(); st.q = +Q.value; render(); });
  $$("#kKernel button").forEach(b => b.addEventListener("click", () => { ch.stop(); st.kernel = b.dataset.v; render(); }));
  const presets = $("#kPresets");
  T6.forEach(r => {
    const b = H("button", { type: "button", class: "chip", "aria-pressed": "false" });
    const lab = () => { b.textContent = (LANG === "tr" ? r[1] : r[0]).replace(" (default)", " ★").replace(" (varsayılan)", " ★"); };
    lab(); onLang(lab);
    b.addEventListener("click", () => { ch.stop(); apply(r[2]); });
    presets.appendChild(b);
  });
  function apply(o) {
    st.kernel = o.kernel; st.q = o.q;
    if (o.p != null) st.p = o.p;
    if (o.sigma != null) st.sigma = o.sigma;
    P.value = st.p; Sg.value = st.sigma; Q.value = st.q; render();
  }
  async function sweep(key, el, seq, ms, ctx) {
    mx.live(true);
    for (const v of seq) {
      const a = st[key];
      await tween(ms, e => { if (!ctx.alive(ctx.token)) return; st[key] = lerp(a, v, e); el.value = st[key]; render(); });
      if (!ctx.alive(ctx.token)) break;
    }
    mx.live(false);
  }
  const ch = Chapter("kernels", [
    { enter: ctx => { apply({ kernel: "power", p: 1, q: 1 }); if (!ctx.instant) sweep("p", P, [0.5, 2.5, 1], 1100, ctx); }, dur: 4400 },
    { enter: ctx => { apply({ kernel: "power", p: 1, q: 1 }); sweep("q", Q, [0, 3, 1], 1100, ctx); }, dur: 4400 },
    { enter: ctx => { apply({ kernel: "gaussian", sigma: 0.25, q: 1 }); sweep("sigma", Sg, [1, 0.5], 1300, ctx); }, dur: 4000 },
    { enter: () => apply({ kernel: "gaussian", sigma: 1, q: 1 }) },
  ]);
  ch.go(0, { instant: true });
  onTheme(render); onLang(render);
}

/* ---------------- 5. Probabilities ---------------- */
function chProb() {
  let mode = "global", showCounts = false, regions = false;
  const tri = { x: 6, y: 7, d: 7 };
  const val = (i, j) => {
    const a = TOY[i][j];
    if (mode === "global") return a / 20;
    return a / tri[campOf(i, j)];
  };
  const mx = MatrixView($("#probMx"), { n: 5, cell: 62, left: 36, top: 36, fs: 15, rowTitle: () => T("i · rating of X", "i · X puanı"), colTitle: () => T("j · rating of Y", "j · Y puanı"),
    onHover: (i, j, ev) => cellTip(i, j, [`a = ${TOY[i][j]}`, `p = ${f(val(i, j), 4)}`], ev), onMove: (i, j, ev) => cellTip(i, j, [`a = ${TOY[i][j]}`, `p = ${f(val(i, j), 4)}`], ev) });
  const barSvg = S("svg", { viewBox: "0 0 400 112", class: "chart", role: "img" });
  $("#probBar").appendChild(barSvg);
  function bar() {
    barSvg.replaceChildren();
    const segs = [["x", 30, T("X-leaning", "X'e yakın")], ["d", 35, T("diagonal", "köşegen")], ["y", 35, T("Y-leaning", "Y'ye yakın")]];
    const col = { x: C["camp-x"], y: C["camp-y"], d: C.diag };
    if (mode === "global") {
      let x0 = 0;
      segs.forEach(([k, pct, lab]) => {
        const w = pct / 100 * 396;
        S("rect", { x: x0 + 2, y: 30, width: w - 2, height: 26, rx: 2, fill: col[k] }, barSvg);
        S("text", { x: x0 + 4, y: 20, class: "axt" }, barSvg).textContent = `${lab} ${pct}%`;
        x0 += w;
      });
      const s = polScores(TOY, WP4);
      S("text", { x: 2, y: 84, "font-size": 12 }, barSvg).textContent = `S_X = ${f(s.SX, 4)}    S_Y = ${f(s.SY, 4)}`;
      S("text", { x: 2, y: 102, "font-size": 11, fill: C.muted }, barSvg).textContent = T("each camp's score includes its share of the sample", "her kampın skoru örneklemdeki payını içerir");
    } else {
      segs.forEach(([k, pct, lab], r) => {
        S("rect", { x: 92, y: 4 + r * 22, width: 300, height: 16, rx: 2, fill: col[k] }, barSvg);
        S("text", { x: 86, y: 12 + r * 22, class: "axt", "text-anchor": "end", "dominant-baseline": "central" }, barSvg).textContent = `${lab} 100%`;
      });
      const sx = 69 / 16 / 6, sy = 85 / 16 / 7;
      S("text", { x: 2, y: 86, "font-size": 12 }, barSvg).textContent = `S_X = ${f(sx, 4)}    S_Y = ${f(sy, 4)}`;
      S("text", { x: 2, y: 104, "font-size": 11, fill: C.muted }, barSvg).textContent = T("= 100 × the per-person scores of chapter 7: size is gone", "= 7. bölümdeki kişi başı skorların 100 katı: büyüklük kayboldu");
    }
    $("#pBarTitle").textContent = mode === "global" ? T("Share of the sample in each region (sums to 100%)", "Her bölgenin örneklem payı (toplam %100)") : T("Each region divided by its own total (each sums to 1)", "Her bölge kendi toplamına bölündü (her biri 1'e toplanır)");
  }
  function render() {
    const V = TOY.map((r, i) => r.map((_, j) => (showCounts ? TOY[i][j] : val(i, j))));
    mx.set(V, { fill: (v, i, j) => ramp("mass", (showCounts ? v / 20 : val(i, j)) / 0.22), text: (v, i, j) => (TOY[i][j] ? (showCounts ? String(v) : fc(v, mode === "global" ? 2 : 3)) : "") });
    mx.regions(regions ? [
      { cells: (i, j) => i > j, color: C["camp-x"] },
      { cells: (i, j) => i < j, color: C["camp-y"] },
      { cells: (i, j) => i === j, color: C.diag },
    ] : []);
    $$("#pMode button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === mode)));
    bar();
  }
  $$("#pMode button").forEach(b => b.addEventListener("click", () => { ch.stop(); mode = b.dataset.v; showCounts = false; regions = true; render(); }));
  const ch = Chapter("prob", [
    { enter: async ctx => { mode = "global"; regions = false; showCounts = !ctx.instant; render(); if (ctx.instant) return; await sleep(900); if (ctx.alive(ctx.token)) { showCounts = false; render(); } } },
    { enter: () => { mode = "global"; showCounts = false; regions = true; render(); } },
    { enter: () => { mode = "tri"; showCounts = false; regions = true; render(); } },
  ]);
  ch.go(0, { instant: true });
  onTheme(render); onLang(render);
}

/* ---------------- 6 & 8. Score trio (W o P = WP) ---------------- */
function ScoreTrio(prefix, type) {
  const W = type === "pol" ? WP4 : WC4;
  const w16 = W.map(r => r.map(v => Math.round(v * 16)));
  const wp = w16.map((r, i) => r.map((v, j) => v * TOY[i][j]));
  const host = $(`#${prefix}Trio`), sumEl = $(`#${prefix}Sum`);
  const views = [];
  const caps = [["W × 16", "W × 16"], ["P × 20", "P × 20"], ["WP × 320", "WP × 320"]];
  caps.forEach((cap, k) => {
    const wrap = H("div");
    wrap.appendChild(H("div", { class: "cap" }, cap[0]));
    const box = H("div"); wrap.appendChild(box);
    host.appendChild(wrap);
    if (k < 2) host.appendChild(H("div", { class: "op" }, k === 0 ? "∘" : "="));
    views.push(MatrixView(box, { n: 5, cell: 30, left: 16, top: 16, fs: 11.5, gap: 1.5,
      onHover: (i, j, ev) => tipFor(i, j, ev), onMove: (i, j, ev) => tipFor(i, j, ev) }));
  });
  function tipFor(i, j, ev) {
    cellTip(i, j, [`w = ${w16[i][j]}/16 = ${f(W[i][j], 4)}`, `p = ${TOY[i][j]}/20 = ${f(TOY[i][j] / 20, 2)}`, `WP = ${wp[i][j]}/320 = ${f(wp[i][j] / 320, 6)}`], ev);
  }
  const wFill = v => (type === "pol" ? ramp("pos", v / 16) : div(v / 16));
  const wpFill = v => (type === "pol" ? ramp("pos", v / 64) : div(v / 48));
  const signed = v => String(v).replace("-", MINUS);
  let keep = null, pinged = [];
  function render() {
    views[0].set(w16, { fill: wFill, text: signed });
    views[1].set(TOY, { fill: v => ramp("mass", v / 4.5), text: v => (v ? String(v) : "") });
    views[2].set(wp, { fill: (v, i, j) => (TOY[i][j] ? wpFill(v) : C.mid), text: (v, i, j) => (TOY[i][j] ? signed(v) : "") });
    views.forEach(v => { v.dim(keep); v.ping(pinged); v.guide(type === "pol" ? "diag" : "anti"); });
  }
  const api = {
    views, wp, w16, render,
    focus(fn) { keep = fn; render(); },
    ping(list) { pinged = list; views.forEach(v => v.ping(list)); },
    sum(html) { sumEl.innerHTML = html; },
  };
  onTheme(render);
  render();
  return api;
}
async function buildSum(trio, region, cls, label, ctx) {
  const cells = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) if (region(i, j) && TOY[i][j]) cells.push([i, j]);
  const terms = [];
  const total320 = cells.reduce((s, [i, j]) => s + trio.wp[i][j], 0);
  const neg = v => String(v).replace("-", MINUS);
  const head = `${label} = (`;
  const end = () => `) / 320 = <b>${neg(total320)}/320</b> = <b>${f(total320 / 320, 6)}</b>`;
  if (ctx.instant || REDUCED) {
    trio.sum(head + cells.map(([i, j]) => `<span class="${cls}">${neg(trio.w16[i][j])}×${TOY[i][j]}</span>`).join(" + ") + end());
    trio.ping([]); return;
  }
  for (const [i, j] of cells) {
    if (!ctx.alive(ctx.token)) return;
    terms.push(`<span class="${cls}">${neg(trio.w16[i][j])}×${TOY[i][j]}</span>`);
    trio.ping([[i, j]]);
    trio.sum(head + terms.join(" + ") + " …");
    await sleep(650);
  }
  if (!ctx.alive(ctx.token)) return;
  trio.ping([]);
  trio.sum(head + terms.join(" + ") + end());
}
function chScores() {
  const trio = ScoreTrio("s", "pol");
  const bars = Bars($("#sBars"), { lo: 0, hi: 0.5 });
  let show = 0;
  function drawBars() {
    const s = polScores(TOY, WP4);
    bars([
      { label: "S_X", value: show >= 1 ? s.SX : 0, color: C["camp-x"], text: show >= 1 ? f(s.SX, 6) : "" },
      { label: "S_Y", value: show >= 2 ? s.SY : 0, color: C["camp-y"], text: show >= 2 ? f(s.SY, 6) : "" },
      { label: "S_pol", value: show >= 3 ? s.Spol : 0, color: C["ink-2"], text: show >= 3 ? f(s.Spol, 5) : "" },
    ]);
  }
  Chapter("scores", [
    { enter: () => { show = 0; trio.focus(null); trio.ping([[0, 4]]); trio.sum(`WP<sub>0,4</sub> = w × p = 16/16 × 4/20 = <b>64/320</b> = <b>0.20</b>`); drawBars(); } },
    { enter: async ctx => { show = 0; drawBars(); trio.focus((i, j) => i > j); await buildSum(trio, (i, j) => i > j, "tx", "S_X", ctx); if (ctx.alive(ctx.token)) { show = 1; drawBars(); } }, dur: 4800 },
    { enter: async ctx => { show = 1; drawBars(); trio.focus((i, j) => i < j); await buildSum(trio, (i, j) => i < j, "ty", "S_Y", ctx); if (ctx.alive(ctx.token)) { show = 2; drawBars(); } }, dur: 4800 },
    { enter: () => { show = 3; trio.focus((i, j) => i !== j); trio.ping([]); trio.sum(`S_pol = S_X + S_Y = <span class="tx">69</span>/320 + <span class="ty">85</span>/320 = <b>154/320</b> = <b>0.48125</b>`); drawBars(); } },
  ]).go(0, { instant: true });
  onTheme(drawBars);
}
function chConsensus() {
  const trio = ScoreTrio("c", "con");
  const bars = Bars($("#cBars"), { lo: -0.25, hi: 0.1, labelW: 70 });
  let show = 0;
  function drawBars() {
    const s = conScores(TOY, WC4);
    bars([
      { label: "S_neg", value: show >= 1 ? s.Sneg : 0, color: C["neg-2"], text: show >= 1 ? f(s.Sneg, 5) : "" },
      { label: "S_pos", value: show >= 2 ? s.Spos : 0, color: C["pos-2"], text: show >= 2 ? f(s.Spos, 2) : "" },
      { label: "S_con", value: show >= 3 ? s.Scon : 0, color: C["ink-2"], text: show >= 3 ? f(s.Scon, 5) : "" },
    ]);
  }
  Chapter("consensus", [
    { enter: () => { show = 0; trio.focus(null); trio.ping([]); trio.sum(T("Anti-diagonal cells (i + j = 4) carry weight 0: 11 respondents contribute nothing.", "Ters köşegen hücreleri (i + j = 4) 0 ağırlık taşır: 11 katılımcı hiçbir katkı yapmaz.")); drawBars(); } },
    { enter: async ctx => { show = 0; drawBars(); trio.focus((i, j) => i + j < 4); await buildSum(trio, (i, j) => i + j < 4, "tx", "S_neg", ctx); if (ctx.alive(ctx.token)) { show = 1; drawBars(); } }, dur: 5200 },
    { enter: async ctx => { show = 1; drawBars(); trio.focus((i, j) => i + j > 4); await buildSum(trio, (i, j) => i + j > 4, "ty", "S_pos", ctx); if (ctx.alive(ctx.token)) { show = 2; drawBars(); } } },
    { enter: () => { show = 3; trio.focus(null); trio.ping([]); trio.sum(`S_con = S_neg + S_pos = ${MINUS}66/320 + 16/320 = <b>${MINUS}50/320</b> = <b>${MINUS}0.15625</b>`); drawBars(); } },
  ]).go(0, { instant: true });
  onTheme(drawBars);
}

/* ---------------- 7. Per person ---------------- */
function chSize() {
  const slider = $("#grow");
  slider.min = 1; slider.max = 8; slider.value = 1;
  let k = 1, view = "sim";
  const dotsHost = $("#sizeDots");
  const dsvg = S("svg", { class: "chart", role: "img", "aria-label": "respondent dots" });
  dotsHost.appendChild(dsvg);
  const raw = Bars($("#sizeRaw"), { lo: 0, hi: 0.65, width: 300, labelW: 40, fmt: v => f(v, 3) });
  const per = Bars($("#sizePer"), { lo: 0, hi: 9, width: 300, labelW: 40, fmt: v => f(v, 2) });
  const paperSvg = S("svg", { class: "chart", role: "img", viewBox: "0 0 420 150" });
  function drawDots() {
    dsvg.replaceChildren();
    const groups = [["x", 6], ["d", 7], ["y", 7 * k]];
    const col = { x: C["camp-x"], y: C["camp-y"], d: C.diag };
    const per = 28, sp = 14.5;
    let idx = 0;
    groups.forEach(([g, cnt]) => { for (let q = 0; q < cnt; q++, idx++) {
      const r = Math.floor(idx / per), cIdx = idx % per;
      S("circle", { cx: 8 + cIdx * sp, cy: 8 + r * sp, r: 5, fill: col[g] }, dsvg);
    } });
    const rows = Math.ceil(idx / per);
    dsvg.setAttribute("viewBox", `0 0 ${8 + per * sp} ${Math.max(3, rows) * sp + 4}`);
  }
  function render() {
    $("#growOut").textContent = "×" + k;
    if (view === "paper") {
      dotsHost.replaceChildren(paperSvg); drawPaper();
      $("#sizeRaw").parentElement.parentElement.hidden = true;
    } else {
      if (!dotsHost.contains(dsvg)) dotsHost.replaceChildren(dsvg);
      $("#sizeRaw").parentElement.parentElement.hidden = false;
      drawDots();
    }
    const A = TOY.map((r, i) => r.map((v, j) => (i < j ? v * k : v)));
    const s = polScores(A, WP4);
    raw([{ label: "S_X", value: s.SX, color: C["camp-x"] }, { label: "S_Y", value: s.SY, color: C["camp-y"] }]);
    per([{ label: "S*_X", value: s.sX * 1e3, color: C["camp-x"] }, { label: "S*_Y", value: s.sY * 1e3, color: C["camp-y"] }]);
    $("#sizeCap").innerHTML = view === "paper"
      ? T("Published bootstrap intervals for the AKP × CHP pair (B = 2,000). Each row has its own scale; the dashed line marks zero.", "AKP × CHP çifti için yayımlanan bootstrap aralıkları (B = 2.000). Her satırın kendi ölçeği var; kesikli çizgi sıfırı gösterir.")
      : `n = ${s.n} · π<sub>X</sub> = ${f(s.piX, 1)} · π<sub>Y</sub> = ${f(s.piY, 1)} · ` + T(`Y-leaning respondents: ${7 * k}`, `Y'ye yakın katılımcı: ${7 * k}`);
  }
  function drawPaper() {
    paperSvg.replaceChildren();
    const rowsP = [
      { lab: "S_X − S_Y", est: -0.053, lo: -0.078, hi: -0.028, d0: -0.1, d1: 0.02, fmt: v => f(v, 3) },
      { lab: "Δ* × 10³", est: 0.02, lo: -0.29, hi: 0.35, d0: -0.6, d1: 0.6, fmt: v => f(v, 2) },
    ];
    rowsP.forEach((r, k2) => {
      const y = 24 + k2 * 66, x = axisLinear(r.d0, r.d1, 110, 400);
      S("text", { x: 100, y, class: "axt", "text-anchor": "end", "dominant-baseline": "central" }, paperSvg).textContent = r.lab;
      S("line", { x1: 110, x2: 400, y1: y + 20, y2: y + 20, class: "axis" }, paperSvg);
      niceTicks(r.d0, r.d1, 4).forEach(tv => { S("text", { x: x(tv), y: y + 33, class: "tick", "text-anchor": "middle" }, paperSvg).textContent = f(tv, Math.abs(r.d1) < 0.1 ? 2 : 1); });
      S("line", { x1: x(0), x2: x(0), y1: y - 14, y2: y + 20, stroke: C.ink, "stroke-dasharray": "3 3" }, paperSvg);
      S("line", { x1: x(r.lo), x2: x(r.hi), y1: y, y2: y, stroke: C["ink-2"], "stroke-width": 2.5 }, paperSvg);
      [r.lo, r.hi].forEach(v => S("line", { x1: x(v), x2: x(v), y1: y - 6, y2: y + 6, stroke: C["ink-2"], "stroke-width": 2 }, paperSvg));
      S("circle", { cx: x(r.est), cy: y, r: 5.5, fill: C.ink, stroke: C.sheet, "stroke-width": 2 }, paperSvg);
      const excl = r.lo > 0 || r.hi < 0;
      S("text", { x: x(r.hi) + 8, y: y - 10, "font-size": 11, fill: C["ink-2"] }, paperSvg).textContent = excl ? T("excludes 0", "0'ı dışlar") : T("contains 0", "0'ı içerir");
    });
  }
  slider.addEventListener("input", () => { ch.stop(); view = "sim"; k = +slider.value; render(); });
  async function grow(ctx) {
    view = "sim";
    for (let v = 1; v <= 8; v++) {
      if (!ctx.alive(ctx.token)) return;
      k = v; slider.value = v; render();
      await sleep(REDUCED ? 0 : 420);
    }
  }
  const ch = Chapter("size", [
    { enter: () => { view = "sim"; k = 1; slider.value = 1; render(); } },
    { enter: () => { view = "sim"; k = 1; slider.value = 1; render(); } },
    { enter: () => { view = "sim"; render(); } },
    { enter: ctx => grow(ctx), dur: 5200 },
    { enter: () => { view = "paper"; render(); } },
  ]);
  ch.go(0, { instant: true });
  onTheme(render); onLang(render);
}

/* ---------------- 9. Bootstrap ---------------- */
function chBoot() {
  /* toy replicate */
  const toks = tokens($("#bootTokens"), TOY_LIST);
  let Ab = TOY.map(r => r.slice()), bN = 0;
  const R = rng(2026);
  const mx = MatrixView($("#bootToyMx"), { n: 5, cell: 44, left: 22, top: 22, fs: 15,
    onHover: (i, j, ev) => cellTip(i, j, [`a* = ${Ab[i][j]}`, `a = ${TOY[i][j]}`], ev), onMove: (i, j, ev) => cellTip(i, j, [`a* = ${Ab[i][j]}`, `a = ${TOY[i][j]}`], ev) });
  const s0 = polScores(TOY, WP4);
  function renderToy() {
    mx.set(Ab, { fill: v => ramp("mass", v / 5), text: v => (v ? String(v) : "") });
    const s = polScores(Ab, WP4);
    RO($("#bootToyRO"), [
      { k: T("original", "orijinal") + " S<sub>pol</sub>", v: f(s0.Spol, 3) },
      { k: (bN ? `b = ${bN} · ` : "") + "S<sub>pol</sub><sup>(b)</sup>", v: bN ? f(s.Spol, 3) : "–", hl: true },
      { k: "S<sub>X</sub><sup>(b)</sup>", v: bN ? f(s.SX, 3) : "–", sw: "x" },
      { k: "S<sub>Y</sub><sup>(b)</sup>", v: bN ? f(s.SY, 3) : "–", sw: "y" },
    ]);
  }
  let drawing = false;
  async function drawReplicate(ctx) {
    if (drawing) return; drawing = true;
    const mult = new Array(20).fill(0);
    toks.forEach(e => { e.classList.add("gone"); e.querySelector(".cnt").textContent = ""; });
    const picks = Array.from({ length: 20 }, () => Math.floor(R() * 20));
    const B = zeros(5);
    for (const p of picks) {
      mult[p]++;
      const e = toks[p]; e.classList.remove("gone"); e.querySelector(".cnt").textContent = mult[p] > 1 ? "×" + mult[p] : "";
      B[TOY_LIST[p].i][TOY_LIST[p].j]++;
      if (!REDUCED) await sleep(70);
    }
    Ab = B; bN++; renderToy();
    drawing = false;
  }
  $("#bootDraw").addEventListener("click", () => { chB.stop(); drawReplicate({ token: 0, alive: () => true }); });

  /* full-size bootstrap */
  const n = total(SIM), cellOf = new Int16Array(n);
  { let k = 0; SIM.forEach((r, i) => r.forEach((c, j) => { for (let q = 0; q < c; q++) cellOf[k++] = i * 11 + j; })); }
  const Wf = W10.flat();
  const pt = polScores(SIM, W10);
  let Bsel = 2000, res = null, running = false, mode = "plain";
  const H1 = S("svg", { class: "chart", role: "img", viewBox: "0 0 540 200", "aria-label": "bootstrap histogram" });
  const H2 = S("svg", { class: "chart", role: "img", viewBox: "0 0 540 150", "aria-label": "difference histogram" });
  $("#bootHist").appendChild(H1); $("#bootDelta").appendChild(H2);
  function replicate(r) {
    const cnt = new Int32Array(121);
    for (let k = 0; k < n; k++) cnt[cellOf[Math.floor(r() * n)]]++;
    let SX = 0, SY = 0, nX = 0, nY = 0;
    for (let i = 0; i < 11; i++) for (let j = 0; j < 11; j++) {
      const a = cnt[i * 11 + j]; if (!a) continue;
      if (i > j) { SX += Wf[i * 11 + j] * a; nX += a; } else if (i < j) { SY += Wf[i * 11 + j] * a; nY += a; }
    }
    SX /= n; SY /= n;
    const sX = SX / (100 * nX / n), sY = SY / (100 * nY / n);
    return [SX + SY, sX * 1e3, sY * 1e3, (sX - sY) * 1e3];
  }
  function summary(arr) {
    const s = Float64Array.from(arr).sort();
    return { mean: arr.reduce((a, b) => a + b, 0) / arr.length, se: sd(arr), lo: quantile(s, 0.025), hi: quantile(s, 0.975) };
  }
  function hist(svg, series, d0, d1, bw, opts = {}) {
    svg.replaceChildren();
    const Wd = 540, Ht = +svg.getAttribute("viewBox").split(" ")[3];
    const top = opts.top || 40, bot = Ht - 24;
    const x = axisLinear(d0, d1, 36, Wd - 10);
    const nb = Math.round((d1 - d0) / bw);
    const counts = series.map(s => { const c = new Array(nb).fill(0); s.data.forEach(v => { const b = Math.floor((v - d0) / bw); if (b >= 0 && b < nb) c[b]++; }); return c; });
    const ymax = Math.max(5, ...counts.flat());
    const y = axisLinear(0, ymax * 1.08, bot, top);
    niceTicks(0, ymax * 1.08, 3).forEach(v => {
      S("line", { x1: 36, x2: Wd - 10, y1: y(v), y2: y(v), class: "gridline" }, svg);
      S("text", { x: 30, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, svg).textContent = v;
    });
    S("line", { x1: 36, x2: Wd - 10, y1: bot, y2: bot, class: "axis" }, svg);
    niceTicks(d0, d1, 6).forEach(v => S("text", { x: x(v), y: bot + 14, class: "tick", "text-anchor": "middle" }, svg).textContent = f(v, 2));
    series.forEach((s, k) => {
      let d = "";
      counts[k].forEach((c, b) => {
        if (c) S("rect", { x: x(d0 + b * bw) + 0.5, y: y(c), width: Math.max(0.5, x(d0 + bw) - x(d0) - 1), height: bot - y(c), fill: s.color, "fill-opacity": series.length > 1 ? 0.38 : 0.55 }, svg);
        d += `${b ? "L" : "M"}${x(d0 + b * bw).toFixed(1)} ${y(c).toFixed(1)}H${x(d0 + (b + 1) * bw).toFixed(1)}`;
      });
      S("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.6 }, svg);
    });
    // hover bins
    const hov = S("rect", { x: 36, y: top, width: Wd - 46, height: bot - top, fill: "transparent" }, svg);
    hov.addEventListener("pointermove", ev => {
      const pt2 = svg.createSVGPoint(); pt2.x = ev.clientX; pt2.y = ev.clientY;
      const loc = pt2.matrixTransform(svg.getScreenCTM().inverse());
      const b = Math.floor((x.inv(loc.x) - d0) / bw);
      if (b < 0 || b >= nb) return hideTip();
      showTip(`<b>${f(d0 + b * bw, 3)} – ${f(d0 + (b + 1) * bw, 3)}</b><br>` + series.map((s, k) => `${s.name}: ${counts[k][b]}`).join("<br>"), ev);
    });
    hov.addEventListener("pointerleave", hideTip);
    // annotations
    series.forEach((s, k) => {
      if (!s.data.length) return;
      const sm = summary(s.data), yy = 12 + k * 13;
      S("line", { x1: x(s.point), x2: x(s.point), y1: top - 4, y2: bot, stroke: s.color, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, svg);
      if (mode === "se" || mode === "ci" || mode === "delta") {
        const [a, b] = mode === "se" ? [sm.mean - sm.se, sm.mean + sm.se] : [sm.lo, sm.hi];
        S("line", { x1: x(a), x2: x(b), y1: yy, y2: yy, stroke: s.color, "stroke-width": mode === "se" ? 4 : 2.2 }, svg);
        [a, b].forEach(v => S("line", { x1: x(v), x2: x(v), y1: yy - 4, y2: yy + 4, stroke: s.color, "stroke-width": 2 }, svg));
        const lab = mode === "se" ? `${s.name}: ±SE = ±${f(sm.se, 3)}` : `${s.name}: [${f(sm.lo, 2)}, ${f(sm.hi, 2)}]`;
        S("text", { x: x(b) + 6, y: yy, "dominant-baseline": "central", "font-size": 10.5, fill: C["ink-2"] }, svg).textContent = lab;
      }
    });
    if (opts.zero) S("line", { x1: x(0), x2: x(0), y1: top - 4, y2: bot, stroke: C.ink, "stroke-width": 1.2 }, svg);
  }
  function renderBig() {
    const d = res || { sX: [], sY: [], delta: [], spol: [] };
    hist(H1, [
      { name: "S*_X", data: d.sX, color: C["camp-x"], point: pt.sX * 1e3 },
      { name: "S*_Y", data: d.sY, color: C["camp-y"], point: pt.sY * 1e3 },
    ], 5.1, 6.3, 0.025);
    hist(H2, [{ name: "Δ*", data: d.delta, color: C["ink-2"], point: (pt.sX - pt.sY) * 1e3 }], -0.8, 0.8, 0.032, { zero: true, top: 30 });
    H2.style.opacity = mode === "delta" || mode === "plain" ? 1 : 0.45;
    $("#bootDeltaTitle").style.opacity = H2.style.opacity;
    const tb = $("#bootTable");
    const rows = [
      ["S<sub>pol</sub>", pt.Spol, d.spol, 3, [0.453, 0.008, 0.436, 0.469]],
      ["S*<sub>X</sub> × 10³", pt.sX * 1e3, d.sX, 2, [5.68, null, 5.43, 5.95]],
      ["S*<sub>Y</sub> × 10³", pt.sY * 1e3, d.sY, 2, [5.66, null, 5.44, 5.88]],
      ["Δ* × 10³", (pt.sX - pt.sY) * 1e3, d.delta, 2, [0.02, null, -0.29, 0.35]],
    ];
    tb.innerHTML = `<thead><tr><th></th><th>${T("estimate", "tahmin")}</th><th>SE</th><th>95% CI</th><th>${T("paper", "makale")}</th></tr></thead><tbody>` +
      rows.map(([lab, est, arr, dd, pub]) => {
        const sm = arr.length > 1 ? summary(arr) : null;
        return `<tr><td class="lbl">${lab}</td><td>${f(est, dd + 1)}</td><td>${sm ? f(sm.se, dd + 1) : "–"}</td><td>${sm ? `[${f(sm.lo, dd)}, ${f(sm.hi, dd)}]` : "–"}</td><td>${f(pub[0], dd)} [${f(pub[2], dd)}, ${f(pub[3], dd)}]</td></tr>`;
      }).join("") + `</tbody>`;
    tb.insertAdjacentHTML("beforeend", `<caption style="caption-side:bottom;text-align:left;font-size:12px;color:var(--muted);padding-top:6px">${d.sX.length ? `B = ${d.sX.length.toLocaleString("en")} · seed 42 · ` : ""}${T("simulated table; paper column from Tables 4 and 6", "simüle tablo; makale sütunu Tablo 4 ve 6'dan")}</caption>`);
  }
  async function run() {
    if (running) return;
    running = true; $("#bootRun").disabled = true;
    const r = rng(42), B = Bsel;
    res = { spol: [], sX: [], sY: [], delta: [] };
    const chunk = REDUCED ? B : 40;
    for (let b = 0; b < B;) {
      for (let q = 0; q < chunk && b < B; q++, b++) { const v = replicate(r); res.spol.push(v[0]); res.sX.push(v[1]); res.sY.push(v[2]); res.delta.push(v[3]); }
      renderBig();
      if (!REDUCED) await new Promise(requestAnimationFrame);
    }
    running = false; $("#bootRun").disabled = false;
  }
  $("#bootRun").addEventListener("click", () => { chB.stop(); run(); });
  $$("#bootB button").forEach(b => b.addEventListener("click", () => {
    Bsel = +b.dataset.v; $$("#bootB button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  }));
  function view(big) { $("#bootToy").hidden = big; $("#bootBig").hidden = !big; }
  const chB = Chapter("boot", [
    { enter: ctx => { view(false); if (ctx.instant) { renderToy(); return; } drawReplicate(ctx); }, dur: 4200 },
    { enter: ctx => { view(false); drawReplicate(ctx); }, dur: 4200 },
    { enter: () => { view(true); mode = "plain"; if (!res || res.sX.length !== Bsel) run(); else renderBig(); }, dur: 5200 },
    { enter: () => { view(true); mode = "se"; if (!res) run(); else renderBig(); } },
    { enter: () => { view(true); mode = "ci"; if (!res) run(); else renderBig(); } },
    { enter: () => { view(true); mode = "delta"; if (!res) run(); else renderBig(); } },
  ]);
  renderToy(); renderBig();
  chB.go(0, { instant: true });
  onTheme(() => { renderToy(); renderBig(); }); onLang(() => { renderToy(); renderBig(); });
}

/* ---------------- 10. Results ---------------- */
function chResults() {
  const tabs = $$("#resTabs [role=tab]");
  const drawn = {};
  const draws = { a: tabA, b: tabB, c: tabC, d: tabD, e: tabE };
  function show(id) {
    tabs.forEach(t => t.setAttribute("aria-selected", String(t.id === "tab-" + id)));
    ["a", "b", "c", "d", "e"].forEach(k => { $("#res-" + k).hidden = k !== id; });
    if (!drawn[id]) { drawn[id] = draws[id](); }
  }
  tabs.forEach(t => t.addEventListener("click", () => show(t.id.slice(4))));
  tabs.forEach((t, k) => t.addEventListener("keydown", ev => {
    if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
    const nx = tabs[(k + (ev.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; nx.focus(); nx.click();
  }));
  show("a");
}
function tabA() {
  let mode = "WP";
  const n = total(SIM), P = SIM.map(r => r.map(v => v / n)), WPm = P.map((r, i) => r.map((v, j) => v * W10[i][j]));
  const pt = polScores(SIM, W10);
  const tipL = (i, j) => [`${T("count", "sayım")} a = ${SIM[i][j]}`, `p = ${f(P[i][j], 4)}`, `w = ${f(W10[i][j], 2)}`, `WP = ${f(WPm[i][j], 4)}`];
  const mx = MatrixView($("#aMx"), { n: 11, cell: 44, left: 40, top: 40, fs: 10.5, rowTitle: () => T("AKP affinity (i)", "AKP yakınlığı (i)"), colTitle: () => T("CHP affinity (j)", "CHP yakınlığı (j)"),
    onHover: (i, j, ev) => cellTip(i, j, tipL(i, j), ev), onMove: (i, j, ev) => cellTip(i, j, tipL(i, j), ev) });
  function render() {
    if (mode === "WP") mx.set(WPm, { fill: (v, i, j) => (SIM[i][j] ? ramp("pos", v / 0.14) : C.sheet), text: (v, i, j) => (SIM[i][j] ? fc(v, 3) : "") });
    else mx.set(SIM, { fill: v => (v ? ramp("mass", Math.sqrt(v / 311)) : C.sheet), text: v => (v ? String(v) : "") });
    mx.guide("diag");
    $$("#aMode button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === mode)));
    const corner = (WPm[0][10] + WPm[10][0]) / pt.Spol;
    RO($("#aRO"), [
      { k: "n", v: n.toLocaleString("en") },
      { k: "π<sub>X</sub> · π<sub>0</sub> · π<sub>Y</sub>", v: `${f(pt.piX, 1)} · ${f(pt.pi0, 1)} · ${f(pt.piY, 1)}` },
      { k: "S<sub>X</sub> (AKP)", v: f(pt.SX, 3), sw: "x" },
      { k: "S<sub>Y</sub> (CHP)", v: f(pt.SY, 3), sw: "y" },
      { k: "S<sub>pol</sub>", v: f(pt.Spol, 3), hl: true, s: T("paper: 0.453", "makale: 0,453") },
      { k: T("two corners", "iki köşe"), v: f(100 * corner, 1) + "%", s: T(`of S_pol, from ${f(100 * 570 / n, 1)}% of people`, `S_pol'un; kişilerin %${f(100 * 570 / n, 1)}'sinden`) },
    ]);
  }
  $$("#aMode button").forEach(b => b.addEventListener("click", () => { mode = b.dataset.v; render(); }));
  render(); onTheme(render); onLang(render);
  return true;
}
function tabB() {
  const K = 13;
  const mx = MatrixView($("#bMx"), { n: K, cell: 38, left: 70, top: 66, fs: 10, rowLabels: T3_ORDER, colLabels: T3_ORDER, rotateCols: true,
    onHover: (i, j, ev) => tipB(i, j, ev), onMove: (i, j, ev) => tipB(i, j, ev) });
  function tipB(i, j, ev) {
    if (i === j) return hideTip();
    const a = T3_ORDER[Math.max(i, j)], b = T3_ORDER[Math.min(i, j)];
    showTip(`<b>${T3_ORDER[i]} × ${T3_ORDER[j]}</b><br>S_pol = ${f(T3[Math.max(i, j)][Math.min(i, j)], 2)}<br>S_con = ${f(T3[Math.min(i, j)][Math.max(i, j)], 2)}`, ev);
  }
  function render() {
    mx.set(T3, {
      fill: (v, i, j) => (i === j ? C.sheet : i > j ? ramp("pos", v / 0.45) : ramp("neg", Math.abs(v) / 0.83)),
      text: (v, i, j) => (i === j ? "" : fc(v, 2)),
    });
    const sel = [...T4.slice(0, 5), ...T4.slice(-5)];
    $("#bTop").innerHTML = `<thead><tr><th>${T("pair", "çift")}</th><th>n</th><th>S<sub>pol</sub></th><th>95% CI</th></tr></thead><tbody>` +
      sel.map((r, k) => `${k === 5 ? `<tr><td colspan="4" style="text-align:center;color:var(--muted)">⋯</td></tr>` : ""}<tr><td class="lbl">${r[0]}</td><td>${r[1]}</td><td>${f(r[4], 3)}</td><td>[${f(r[5], 3)}, ${f(r[6], 3)}]</td></tr>`).join("") + "</tbody>";
  }
  render(); onTheme(render); onLang(render);
  return true;
}
function tabC() {
  const svg = S("svg", { class: "chart", role: "img", viewBox: "0 0 540 360", "aria-label": "salience scatter" });
  $("#cScatter").appendChild(svg);
  const xs = PARTIES.map(p => p.mean), ys = PARTIES.map(p => p.spol);
  const mxv = xs.reduce((a, b) => a + b) / xs.length, myv = ys.reduce((a, b) => a + b) / ys.length;
  let sxy = 0, sxx = 0, syy = 0;
  xs.forEach((x, k) => { sxy += (x - mxv) * (ys[k] - myv); sxx += (x - mxv) ** 2; syy += (ys[k] - myv) ** 2; });
  const b = sxy / sxx, a = myv - b * mxv, r = sxy / Math.sqrt(sxx * syy);
  $("#cR").textContent = `r = ${f(r, 2)}`; $("#cR2").textContent = `r = ${f(r, 2).replace(".", ",")}`;
  const off = { GP: [7, 11, "start"], DEVA: [0, -11, "middle"], HÜDAPAR: [-8, 2, "end"], SP: [2, 16, "middle"], MP: [7, 12, "start"], "İYİ": [7, 9, "start"], "TİP": [-7, -7, "end"],
    DEM: [-8, -6, "end"], ZP: [0, -12, "middle"], YRP: [8, 3, "start"], MHP: [8, 4, "start"], AKP: [8, 4, "start"], CHP: [-8, -6, "end"] };
  function render() {
    svg.replaceChildren();
    const X = axisLinear(0.3, 4.4, 50, 528), Y = axisLinear(0.095, 0.32, 324, 14);
    niceTicks(0.1, 0.3, 4).forEach(v => {
      S("line", { x1: 50, x2: 528, y1: Y(v), y2: Y(v), class: "gridline" }, svg);
      S("text", { x: 44, y: Y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, svg).textContent = f(v, 2);
    });
    [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4].forEach(v => S("text", { x: X(v), y: 340, class: "tick", "text-anchor": "middle" }, svg).textContent = v);
    S("text", { x: 289, y: 357, class: "axt", "text-anchor": "middle" }, svg).textContent = T("mean affinity rating of the party (0–10)", "partinin ortalama yakınlık puanı (0–10)");
    S("text", { x: 12, y: 170, class: "axt", "text-anchor": "middle", transform: "rotate(-90 12 170)" }, svg).textContent = T("mean S_pol against the other 12", "diğer 12'ye karşı ortalama S_pol");
    S("line", { x1: X(0.4), y1: Y(a + b * 0.4), x2: X(4.25), y2: Y(a + b * 4.25), stroke: C.muted, "stroke-width": 1.5, "stroke-dasharray": "6 4" }, svg);
    PARTIES.forEach(p => {
      const yh = a + b * p.mean, res = p.spol - yh;
      S("line", { x1: X(p.mean), x2: X(p.mean), y1: Y(yh), y2: Y(p.spol), stroke: res > 0 ? C["pos-2"] : C["neg-2"], "stroke-width": 2.5 }, svg);
    });
    PARTIES.forEach(p => {
      const c = S("circle", { cx: X(p.mean), cy: Y(p.spol), r: 5.5, fill: C.ink, stroke: C.sheet, "stroke-width": 2 }, svg);
      const o = off[p.a] || [7, 4, "start"];
      S("text", { x: X(p.mean) + o[0], y: Y(p.spol) + o[1], "text-anchor": o[2], "font-size": 11, "font-weight": 600, "font-family": "Archivo, system-ui, sans-serif", fill: C["ink-2"] }, svg).textContent = p.a;
      const hit = S("circle", { cx: X(p.mean), cy: Y(p.spol), r: 12, fill: "transparent" }, svg);
      const tipT = ev => showTip(`<b>${p.a}</b> · ${LANG === "tr" ? p.tr : p.en}<br>${T("mean affinity", "ort. yakınlık")} ${f(p.mean, 2)}<br>S̄_pol ${f(p.spol, 3)} · ${T("fit", "uyum")} ${f(a + b * p.mean, 3)}<br>${T("residual", "artık")} ${p.spol - (a + b * p.mean) >= 0 ? "+" : ""}${f(p.spol - (a + b * p.mean), 3)}`, ev);
      hit.addEventListener("pointerenter", tipT); hit.addEventListener("pointermove", tipT); hit.addEventListener("pointerleave", hideTip);
    });
  }
  render(); onTheme(render); onLang(render);
  return true;
}
function tabD() {
  const svg = S("svg", { class: "chart", role: "img", viewBox: "0 0 540 300", "aria-label": "kernel sensitivity" });
  $("#dChart").appendChild(svg);
  const sims = T6.map(r => polScores(SIM, weights(10, r[2]).W).Spol);
  function render() {
    svg.replaceChildren();
    const X = axisLinear(0.36, 0.56, 190, 528), row = 32, top = 14;
    niceTicks(0.36, 0.56, 5).forEach(v => {
      S("line", { x1: X(v), x2: X(v), y1: top - 6, y2: top + T6.length * row, class: "gridline" }, svg);
      S("text", { x: X(v), y: top + T6.length * row + 14, class: "tick", "text-anchor": "middle" }, svg).textContent = f(v, 2);
    });
    T6.forEach((r, k) => {
      const y = top + k * row + row / 2;
      S("text", { x: 182, y, class: "axt", "text-anchor": "end", "dominant-baseline": "central" }, svg).textContent = LANG === "tr" ? r[1] : r[0];
      S("line", { x1: X(r[5]), x2: X(r[6]), y1: y, y2: y, stroke: C["ink-2"], "stroke-width": 2.2 }, svg);
      S("circle", { cx: X(r[4]), cy: y, r: 5, fill: C.ink, stroke: C.sheet, "stroke-width": 1.5 }, svg);
      const s = sims[k], d = 5.5;
      S("path", { d: `M${X(s)} ${y - d - 7}l${d} ${d}l${-d} ${d}l${-d} ${-d}z`, fill: C.sheet, stroke: C.pen, "stroke-width": 2 }, svg);
      const hit = S("rect", { x: 0, y: y - row / 2, width: 540, height: row, fill: "transparent" }, svg);
      const tt = ev => showTip(`<b>${LANG === "tr" ? r[1] : r[0]}</b><br>${T("published", "yayımlanan")} ${f(r[4], 3)} [${f(r[5], 3)}, ${f(r[6], 3)}]<br>${T("this page (simulated)", "bu sayfa (simüle)")} ${f(s, 3)}<br>ρ = ${f(r[10], 4)}`, ev);
      hit.addEventListener("pointerenter", tt); hit.addEventListener("pointermove", tt); hit.addEventListener("pointerleave", hideTip);
    });
    S("text", { x: X(0.46), y: 298, class: "axt", "text-anchor": "middle" }, svg).textContent = "S_pol, AKP × CHP";
    $("#dTable").innerHTML = `<thead><tr><th>${T("specification", "tanım")}</th><th>max|m|</th><th>${T("78 pairs: mean", "78 çift: ort.")}</th><th>ρ</th></tr></thead><tbody>` +
      T6.map(r => `<tr><td class="lbl">${LANG === "tr" ? r[1] : r[0]}</td><td>${f(r[3], 4)}</td><td>${f(r[7], 3)}</td><td>${f(r[10], 4)}</td></tr>`).join("") + "</tbody>";
  }
  render(); onTheme(render); onLang(render);
  return true;
}
function jacobiEigen(A) {
  const n = A.length, a = A.map(r => r.slice()), V = A.map((r, i) => r.map((_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  return { values: a.map((r, i) => r[i]), vectors: V };
}
function tabE() {
  const K = 13, D = zeros(K);
  for (let i = 0; i < K; i++) for (let j = 0; j < i; j++) { D[i][j] = D[j][i] = T3[i][j]; }
  const D2 = D.map(r => r.map(v => v * v));
  const rowM = D2.map(r => r.reduce((a, b) => a + b) / K), allM = rowM.reduce((a, b) => a + b) / K;
  const B = D2.map((r, i) => r.map((v, j) => -0.5 * (v - rowM[i] - rowM[j] + allM)));
  const { values, vectors } = jacobiEigen(B);
  const order = values.map((v, k) => k).sort((p, q) => values[q] - values[p]);
  const pos = values.filter(v => v > 1e-12).reduce((a, b) => a + b, 0);
  const share = (values[order[0]] + values[order[1]]) / pos;
  let XY = T3_ORDER.map((_, i) => [0, 1].map(d => vectors[i][order[d]] * Math.sqrt(values[order[d]])));
  const ci = T3_ORDER.indexOf("CHP");
  if (XY[ci][0] < 0) XY = XY.map(p => [-p[0], p[1]]);
  if (XY[ci][1] < 0) XY = XY.map(p => [p[0], -p[1]]);
  const svg = S("svg", { class: "chart", role: "img", viewBox: "0 0 540 380", "aria-label": "party map" });
  $("#eMap").appendChild(svg);
  const links = T4.filter(r => r[4] > 0.29).map(r => r[0].split(" × "));
  const meanOf = a => PARTIES.find(p => p.a === a).mean;
  const off = { "İYİ": [-4, -11, "end"], ZP: [8, -6, "start"], GP: [-8, 2, "end"], MP: [8, 4, "start"], DEVA: [8, 12, "start"], SP: [-8, 2, "end"], "TİP": [8, 4, "start"], YRP: [-9, 3, "end"],
    HÜDAPAR: [0, 18, "middle"], DEM: [9, 4, "start"], AKP: [9, -8, "start"], MHP: [8, -9, "start"], CHP: [-10, -12, "end"] };
  function render() {
    svg.replaceChildren();
    const X = axisLinear(-0.25, 0.27, 52, 528), Y = axisLinear(-0.13, 0.16, 344, 14);
    [-0.2, -0.1, 0, 0.1, 0.2].forEach(v => S("text", { x: X(v), y: 360, class: "tick", "text-anchor": "middle" }, svg).textContent = f(v, 1));
    [-0.1, 0, 0.1].forEach(v => S("text", { x: 46, y: Y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, svg).textContent = f(v, 1));
    S("line", { x1: X(0), x2: X(0), y1: 10, y2: 348, class: "gridline" }, svg);
    S("line", { x1: 52, x2: 528, y1: Y(0), y2: Y(0), class: "gridline" }, svg);
    S("text", { x: 284, y: 377, class: "axt", "text-anchor": "middle" }, svg).textContent = T("first principal coordinate", "birinci temel koordinat");
    S("text", { x: 10, y: 180, class: "axt", "text-anchor": "middle", transform: "rotate(-90 10 180)" }, svg).textContent = T("second principal coordinate", "ikinci temel koordinat");
    links.forEach(([p, q]) => {
      const a = XY[T3_ORDER.indexOf(p)], b = XY[T3_ORDER.indexOf(q)];
      S("line", { x1: X(a[0]), y1: Y(a[1]), x2: X(b[0]), y2: Y(b[1]), stroke: C["pos-2"], "stroke-width": 1.2, opacity: 0.75 }, svg);
    });
    T3_ORDER.forEach((p, i) => {
      const [x, y] = XY[i], rr = 2.5 + 3.2 * Math.sqrt(meanOf(p));
      S("circle", { cx: X(x), cy: Y(y), r: rr, fill: C.pen, "fill-opacity": 0.75, stroke: C.sheet, "stroke-width": 1.5 }, svg);
      const o = off[p] || [8, 4, "start"];
      S("text", { x: X(x) + o[0], y: Y(y) + o[1], "text-anchor": o[2], "font-size": 11.5, "font-weight": 700, "font-family": "Archivo, system-ui, sans-serif", fill: C.ink }, svg).textContent = p;
      const hit = S("circle", { cx: X(x), cy: Y(y), r: 13, fill: "transparent" }, svg);
      const tt = ev => showTip(`<b>${p}</b><br>(${f(x, 3)}, ${f(y, 3)})<br>${T("mean affinity", "ort. yakınlık")} ${f(meanOf(p), 2)}`, ev);
      hit.addEventListener("pointerenter", tt); hit.addEventListener("pointermove", tt); hit.addEventListener("pointerleave", hideTip);
    });
    $("#eCap").innerHTML = T(`First two axes recover <b>${f(100 * share, 1)}%</b> of the configuration (paper: 81.5%). AKP sits at ${f(XY[0][0], 3)} and CHP at ${f(XY[1][0], 3)} on the first axis. Red lines: pairs with S<sub>pol</sub> &gt; 0.29 (Table 4).`,
      `İlk iki eksen yapılanmanın <b>%${f(100 * share, 1)}</b>'ini açıklıyor (makale: %81,5). Birinci eksende AKP ${f(XY[0][0], 3)}, CHP ${f(XY[1][0], 3)} konumunda. Kırmızı çizgiler: S<sub>pol</sub> &gt; 0,29 olan çiftler (Tablo 4).`);
  }
  render(); onTheme(render); onLang(render);
  return true;
}

/* ---------------- TOC highlight ---------------- */
function tocSpy() {
  const links = $$("#toc a"), map = new Map(links.map(a => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      links.forEach(a => a.classList.remove("on"));
      const a = map.get(e.target.id);
      if (a) { a.classList.add("on"); const t = $("#toc"); t.scrollTo({ left: a.offsetLeft - t.clientWidth / 2 + a.offsetWidth / 2, behavior: REDUCED ? "auto" : "smooth" }); }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  $$("section.chapter").forEach(s => io.observe(s));
}

/* ---------------- init ---------------- */
function init() {
  readPalette(); watchTheme();
  let saved = null;
  try { saved = localStorage.getItem("bipol-lang"); } catch (e) { /* storage unavailable */ }
  setLang(saved || ((navigator.language || "").toLowerCase().startsWith("tr") ? "tr" : "en"), false);
  $$(".lang button").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));
  const hero = Hero();
  const run = (name, fn) => { try { fn(); } catch (e) { console.error(name, e); } };
  run("problem", chProblem); run("table", chTable); run("weights", chWeights); run("kernels", chKernels);
  run("prob", chProb); run("scores", chScores); run("size", chSize); run("consensus", chConsensus);
  run("boot", chBoot); run("results", chResults); run("toc", tocSpy);
  hero.play();
}
init();
