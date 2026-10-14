"use strict";
/* ============================================================
   Core: utilities, language, colour, maths, model, components
   ============================================================ */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const SVGNS = "http://www.w3.org/2000/svg";
function S(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function H(tag, attrs = {}, html) {
  const e = document.createElement(tag);
  for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (html != null) e.innerHTML = html;
  return e;
}
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const MINUS = "−";
function f(x, d = 3) {
  if (!isFinite(x)) return "–";
  const s = Math.abs(x).toFixed(d);
  return (x < 0 && +s !== 0 ? MINUS : "") + s;
}
/* ".45" style for compact cells */
function fc(x, d = 2) {
  const s = f(x, d);
  return s.replace(/^(−?)0\./, "$1.");
}
function tween(dur, fn) {
  return new Promise(res => {
    if (REDUCED || dur <= 0) { fn(1); res(); return; }
    const t0 = performance.now();
    const tick = now => {
      const t = clamp((now - t0) / dur, 0, 1);
      fn(ease(t));
      if (t < 1) requestAnimationFrame(tick); else res();
    };
    requestAnimationFrame(tick);
  });
}

/* ---------- language ---------- */
let LANG = "en";
const langHooks = [];
const T = (en, tr) => (LANG === "tr" ? tr : en);
function onLang(fn) { langHooks.push(fn); }
function setLang(l, save = true) {
  LANG = l;
  $("#app").dataset.lang = l;
  document.documentElement.lang = l;
  $$(".lang button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
  if (save) { try { localStorage.setItem("bipol-lang", l); } catch (e) { /* storage unavailable */ } }
  langHooks.forEach(fn => fn());
}

/* ---------- colour (OKLab mixing, theme tokens) ---------- */
function parseColor(c) {
  c = c.trim();
  if (c[0] === "#") {
    if (c.length === 4) c = "#" + [...c.slice(1)].map(x => x + x).join("");
    return [1, 3, 5].map(k => parseInt(c.slice(k, k + 2), 16));
  }
  const m = c.match(/[\d.]+/g);
  return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
}
const toLin = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
const fromLin = v => 255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
function rgb2lab([r, g, b]) {
  r = toLin(r); g = toLin(g); b = toLin(b);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
          1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
          0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function lab2rgb([L, a, b]) {
  const l = Math.pow(L + 0.3963377774 * a + 0.2158037573 * b, 3);
  const m = Math.pow(L - 0.1055613458 * a - 0.0638541728 * b, 3);
  const s = Math.pow(L - 0.0894841775 * a - 1.291485548 * b, 3);
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
          -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
          -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s].map(v => clamp(Math.round(fromLin(v)), 0, 255));
}
function mixLab(A, B, t) { return [0, 1, 2].map(k => lerp(A[k], B[k], t)); }
const rgbStr = c => `rgb(${c[0]},${c[1]},${c[2]})`;
function luminance([r, g, b]) { return 0.2126 * toLin(r) + 0.7152 * toLin(g) + 0.0722 * toLin(b); }
function contrast(a, b) { const x = luminance(a), y = luminance(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); }

const C = {};               // resolved theme colours
const themeHooks = [];
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim() || "#888";
  for (const k of ["paper", "sheet", "ink", "ink-2", "muted", "rule", "grid", "pen", "camp-x", "camp-y", "diag",
    "neg-3", "neg-2", "neg-1", "mid", "pos-1", "pos-2", "pos-3"]) C[k] = v("--" + k);
  const lab = k => rgb2lab(parseColor(C[k]));
  C.lab = {
    pos: [lab("mid"), lab("pos-1"), lab("pos-2"), lab("pos-3")],
    neg: [lab("mid"), lab("neg-1"), lab("neg-2"), lab("neg-3")],
    mass: [lab("mid"), mixLab(lab("mid"), lab("ink-2"), 0.4), lab("ink-2"), lab("ink")],
  };
  C.inkRGB = parseColor(C.ink); C.sheetRGB = parseColor(C.sheet);
}
function rampRGB(stops, t) {
  t = clamp(t, 0, 1) * (stops.length - 1);
  const k = Math.min(Math.floor(t), stops.length - 2);
  return lab2rgb(mixLab(stops[k], stops[k + 1], t - k));
}
const ramp = (name, t) => rgbStr(rampRGB(C.lab[name], t));
/* diverging: v in [-1, 1] */
const div = v => v >= 0 ? ramp("pos", v) : ramp("neg", -v);
function textOn(fill) {
  const c = parseColor(fill);
  return contrast(c, C.inkRGB) >= contrast(c, C.sheetRGB) ? C.ink : C.sheet;
}
function onTheme(fn) { themeHooks.push(fn); }
function watchTheme() {
  const fire = () => { readPalette(); themeHooks.forEach(fn => fn()); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fire);
  new MutationObserver(fire).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
}

/* ---------- MathJax helpers ---------- */
const pendingTex = new Set();
function tex(el, src, display = false) {
  el.dataset.tex = src;
  el.dataset.display = display ? "1" : "";
  if (window.MathJax && MathJax.tex2svg && window.__mjReady) {
    try { el.replaceChildren(MathJax.tex2svg(src, { display })); pendingTex.delete(el); return; } catch (e) { /* fall through */ }
  }
  const s = H("span", { class: "tex-fallback" }); s.textContent = src;
  el.replaceChildren(s);
  pendingTex.add(el);
}
function flushTex() { [...pendingTex].forEach(el => tex(el, el.dataset.tex, !!el.dataset.display)); }

/* ---------- tooltip ---------- */
const tip = { el: null };
function showTip(html, ev) {
  const t = tip.el || (tip.el = $("#tip"));
  t.innerHTML = html; t.classList.add("on");
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  t.style.left = Math.max(8, x) + "px"; t.style.top = Math.max(8, y) + "px";
}
function hideTip() { tip.el && tip.el.classList.remove("on"); }
addEventListener("scroll", hideTip, { passive: true });

/* ============================================================
   Model: mirrors polarization/weights.py and scores.py
   ============================================================ */
function rawWeights(N, { type = "polarization", kernel = "power", p = 1, q = 1, sigma = 0.5 } = {}) {
  const M = [];
  for (let i = 0; i <= N; i++) {
    M.push([]);
    for (let j = 0; j <= N; j++) {
      let d, a;
      if (type === "polarization") { d = Math.abs(i - j) / N; a = 1 - Math.abs(i + j - N) / N; }
      else { d = (i + j) / N - 1; a = (N - Math.abs(i - j)) / N; }
      let g;
      if (kernel === "power") g = Math.pow(d, p);
      else g = (1 - Math.exp(-(d * d) / (2 * sigma * sigma))) * Math.sign(d);
      M[i].push(g * Math.pow(a, q));
    }
  }
  return M;
}
function weights(N, opts) {
  const M = rawWeights(N, opts);
  let mx = 0;
  M.forEach(r => r.forEach(v => { mx = Math.max(mx, Math.abs(v)); }));
  const W = M.map(r => r.map(v => (mx > 0 ? v / mx : v)));
  return { M, W, max: mx };
}
const total = A => A.reduce((s, r) => s + r.reduce((a, b) => a + b, 0), 0);
function polScores(A, W) {
  const n = total(A), K = A.length;
  let SX = 0, SY = 0, nX = 0, nY = 0, n0 = 0;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    const a = A[i][j], wp = W[i][j] * a / n;
    if (i > j) { SX += wp; nX += a; } else if (i < j) { SY += wp; nY += a; } else n0 += a;
  }
  const piX = 100 * nX / n, piY = 100 * nY / n;
  return { n, SX, SY, Spol: SX + SY, nX, nY, n0, piX, piY, pi0: 100 * n0 / n,
    sX: piX > 0 ? SX / piX : 0, sY: piY > 0 ? SY / piY : 0 };
}
function conScores(A, W) {
  const n = total(A), K = A.length, N = K - 1;
  let Sneg = 0, Spos = 0, nneg = 0, npos = 0, nanti = 0;
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    const a = A[i][j], wp = W[i][j] * a / n;
    if (i + j < N) { Sneg += wp; nneg += a; } else if (i + j > N) { Spos += wp; npos += a; } else nanti += a;
  }
  return { n, Sneg, Spos, Scon: Sneg + Spos, nneg, npos, nanti };
}
/* seeded PRNG (mulberry32) */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function sd(arr) {
  const m = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - m) * (b - m), 0) / (arr.length - 1));
}
const zeros = K => Array.from({ length: K }, () => new Array(K).fill(0));

/* ============================================================
   MatrixView: an SVG heat-table with regions, guides, hover
   ============================================================ */
function MatrixView(host, o) {
  const n = o.n, c = o.cell || 30, L = o.left ?? 30, Tp = o.top ?? 30, gap = o.gap ?? 1.5;
  const Wd = L + n * c + (o.right ?? 4), Ht = Tp + n * c + (o.bottom ?? 4);
  const svg = S("svg", { viewBox: `0 0 ${Wd} ${Ht}`, class: "mx", role: "img", "aria-label": o.label || "matrix" });
  const gTicks = S("g", {}, svg), gCells = S("g", {}, svg), gOver = S("g", {}, svg);
  const rowLab = o.rowLabels || [...Array(n).keys()], colLab = o.colLabels || [...Array(n).keys()];
  const fs = o.fs || Math.max(7, c * 0.36);
  if (o.ticks !== false) {
    rowLab.forEach((t, k) => { const e = S("text", { x: L - 5, y: Tp + k * c + c / 2, class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, gTicks); e.textContent = t; });
    colLab.forEach((t, k) => {
      const x = L + k * c + c / 2, y = Tp - 5;
      const e = o.rotateCols
        ? S("text", { x, y, class: "tick", "text-anchor": "start", transform: `rotate(-55 ${x} ${y})` }, gTicks)
        : S("text", { x, y, class: "tick", "text-anchor": "middle" }, gTicks);
      e.textContent = t;
    });
  }
  const axT = {};
  if (o.rowTitle != null) { axT.row = S("text", { class: "axt", "text-anchor": "middle", transform: `translate(${o.rowTitleX ?? 9} ${Tp + n * c / 2}) rotate(-90)` }, gTicks); }
  if (o.colTitle != null) { axT.col = S("text", { class: "axt", "text-anchor": "middle", x: L + n * c / 2, y: o.colTitleY ?? 10 }, gTicks); }
  function titles() {
    if (axT.row) axT.row.textContent = typeof o.rowTitle === "function" ? o.rowTitle() : o.rowTitle;
    if (axT.col) axT.col.textContent = typeof o.colTitle === "function" ? o.colTitle() : o.colTitle;
  }
  titles(); onLang(titles);
  const cells = [];
  for (let i = 0; i < n; i++) {
    cells.push([]);
    for (let j = 0; j < n; j++) {
      const g = S("g", { class: "cell" }, gCells);
      const r = S("rect", { x: L + j * c + gap / 2, y: Tp + i * c + gap / 2, width: c - gap, height: c - gap, rx: 1.5 }, g);
      const t = S("text", { x: L + j * c + c / 2, y: Tp + i * c + c / 2, "font-size": fs }, g);
      g.addEventListener("pointerenter", ev => o.onHover && o.onHover(i, j, ev));
      g.addEventListener("pointermove", ev => o.onMove && o.onMove(i, j, ev));
      g.addEventListener("click", ev => o.onClick && o.onClick(i, j, ev));
      cells[i].push({ g, r, t });
    }
  }
  svg.addEventListener("pointerleave", () => { hideTip(); o.onLeave && o.onLeave(); });
  host.appendChild(svg);
  const api = {
    svg, n, c, L, T: Tp, cells,
    x: j => L + j * c, y: i => Tp + i * c,
    set(vals, { fill, text } = {}) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = vals[i][j], { r, t } = cells[i][j];
        const fl = fill ? fill(v, i, j) : C.mid;
        r.style.fill = fl;
        const s = text ? text(v, i, j) : "";
        t.textContent = s;
        t.style.fill = textOn(fl);
      }
    },
    dim(keep) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cells[i][j].g.classList.toggle("dim", !!keep && !keep(i, j));
    },
    ping(list) {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cells[i][j].g.classList.remove("ping");
      (list || []).forEach(([i, j]) => cells[i][j].g.classList.add("ping"));
    },
    regions(list) {
      gOver.querySelectorAll(".region,.rlabel").forEach(e => e.remove());
      (list || []).forEach(rg => {
        let d = "";
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          if (!rg.cells(i, j)) continue;
          const x0 = L + j * c, y0 = Tp + i * c, x1 = x0 + c, y1 = y0 + c;
          if (i === 0 || !rg.cells(i - 1, j)) d += `M${x0} ${y0}H${x1}`;
          if (i === n - 1 || !rg.cells(i + 1, j)) d += `M${x0} ${y1}H${x1}`;
          if (j === 0 || !rg.cells(i, j - 1)) d += `M${x0} ${y0}V${y1}`;
          if (j === n - 1 || !rg.cells(i, j + 1)) d += `M${x1} ${y0}V${y1}`;
        }
        S("path", { d, class: "region", stroke: rg.color, "stroke-linecap": "square" }, gOver);
        if (rg.label) {
          const [li, lj] = rg.at;
          const t = S("text", { x: L + lj * c + c / 2, y: Tp + li * c + c / 2, class: "rlabel", "text-anchor": rg.anchor || "middle", "dominant-baseline": "central", fill: rg.textColor || C.ink }, gOver);
          t.textContent = rg.label;
        }
      });
    },
    guide(kind) {
      gOver.querySelectorAll(".guide").forEach(e => e.remove());
      if (kind === "diag") S("line", { x1: L, y1: Tp, x2: L + n * c, y2: Tp + n * c, class: "guide" }, gOver);
      if (kind === "anti") S("line", { x1: L + n * c, y1: Tp, x2: L, y2: Tp + n * c, class: "guide" }, gOver);
    },
    live(on) { svg.classList.toggle("live", on); },
  };
  return api;
}

/* ---------- small chart helpers ---------- */
function axisLinear(d0, d1, r0, r1) {
  const s = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
  s.inv = px => d0 + (px - r0) / (r1 - r0) * (d1 - d0);
  return s;
}
function niceTicks(lo, hi, count = 5) {
  const span = hi - lo, step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= count) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
/* horizontal bar list: rows [{label, value, color, note}] */
function HBars(host, { max, width = 360, row = 26, labelW = 70, fmt = v => f(v, 3) }) {
  const svg = S("svg", { class: "chart", role: "img" });
  host.appendChild(svg);
  return {
    svg,
    draw(rows, mx = max) {
      svg.replaceChildren();
      const Ht = rows.length * row + 4;
      svg.setAttribute("viewBox", `0 0 ${width} ${Ht}`);
      const x = axisLinear(0, mx, labelW, width - 58);
      rows.forEach((r, k) => {
        const y = k * row + 4;
        const lab = S("text", { x: labelW - 8, y: y + row / 2 - 2, "text-anchor": "end", "dominant-baseline": "central", class: "axt" }, svg);
        lab.textContent = r.label;
        S("rect", { x: labelW, y: y + 3, width: Math.max(0, x(Math.min(r.value, mx)) - labelW), height: row - 10, rx: 2, fill: r.color }, svg);
        const v = S("text", { x: x(Math.min(r.value, mx)) + 6, y: y + row / 2 - 2, "dominant-baseline": "central", "font-size": 11.5 }, svg);
        v.textContent = r.text != null ? r.text : fmt(r.value);
      });
    },
  };
}

/* ============================================================
   Chapter controller: step list + stage footer + autoplay
   ============================================================ */
const ICON = {
  prev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.6 2.2 4.8 8l5.8 5.8 1.2-1.2L7.2 8l4.6-4.6z"/></svg>',
  next: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5.4 2.2 11.2 8l-5.8 5.8-1.2-1.2L8.8 8 4.2 3.4z"/></svg>',
  play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>',
  pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zm5 0h3v11H9z"/></svg>',
};
const chapters = {};
function Chapter(name, steps) {
  const list = $(`.steps[data-chapter="${name}"]`), stage = $(`.stage[data-chapter="${name}"]`);
  const items = $$(".step", list);
  const foot = H("div", { class: "stage-foot" });
  const prev = H("button", { type: "button", class: "icon-btn" }, ICON.prev);
  const where = H("div", { class: "where", "aria-live": "polite" });
  const next = H("button", { type: "button", class: "icon-btn" }, ICON.next);
  const play = H("button", { type: "button", class: "icon-btn play-btn" });
  foot.append(prev, where, next, play);
  stage.appendChild(foot);
  let cur = -1, playing = false, timer = null, token = 0;
  const title = k => { const h = items[k].querySelector(`h3 [lang="${LANG}"]`) || items[k].querySelector("h3"); return h.textContent.trim(); };
  function paint() {
    items.forEach((li, i) => { li.classList.toggle("on", i === cur); li.querySelector("h3 button").setAttribute("aria-current", i === cur ? "step" : "false"); });
    where.innerHTML = `<b>${cur + 1}/${items.length}</b>${title(cur)}`;
    prev.disabled = cur <= 0; next.disabled = cur >= items.length - 1;
    prev.setAttribute("aria-label", T("Previous step", "Önceki adım"));
    next.setAttribute("aria-label", T("Next step", "Sonraki adım"));
    play.innerHTML = (playing ? ICON.pause : ICON.play) + `<span class="t">${playing ? T("Pause", "Duraklat") : T("Play", "Oynat")}</span>`;
    play.setAttribute("aria-label", playing ? T("Pause", "Duraklat") : T("Play all steps", "Tüm adımları oynat"));
  }
  function go(k, opts = {}) {
    k = clamp(k, 0, items.length - 1);
    const from = cur; cur = k; token++;
    paint();
    steps[k].enter({ from, instant: !!opts.instant, token, alive: t => t === token });
  }
  function stop() { playing = false; clearTimeout(timer); paint(); }
  function schedule() {
    clearTimeout(timer);
    if (!playing) return;
    if (cur >= items.length - 1) { stop(); return; }
    timer = setTimeout(() => { go(cur + 1); schedule(); }, REDUCED ? 2600 : (steps[cur].dur || 3800));
  }
  items.forEach((li, i) => li.addEventListener("click", ev => { if (ev.target.closest("a")) return; stop(); if (i !== cur) go(i); }));
  prev.addEventListener("click", () => { stop(); go(cur - 1); });
  next.addEventListener("click", () => { stop(); go(cur + 1); });
  play.addEventListener("click", () => {
    if (playing) { stop(); return; }
    playing = true;
    if (cur >= items.length - 1) go(0); else if (cur < 0) go(0);
    paint(); schedule();
  });
  onLang(paint);
  const api = { go, stop, get cur() { return cur; }, steps };
  chapters[name] = api;
  return api;
}

/* ============================================================
   Hero: 2,218 simulated respondents fall into the AKP x CHP table
   ============================================================ */
function Hero() {
  const cv = $("#heroCanvas"), ctx = cv.getContext("2d");
  const N = 10, K = 11, Wt = weights(N).W;
  let dots = [], geo = null, t0 = 0, raf = 0, prog = REDUCED ? 99 : 0;
  const R = rng(7);
  const gauss = () => { let u = 0, v = 0; while (!u) u = R(); while (!v) v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  // stable dot list
  const base = [];
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    const c = SIM[i][j], side = Math.ceil(Math.sqrt(c));
    for (let k = 0; k < c; k++) base.push({ i, j, k, side, camp: i > j ? "x" : i < j ? "y" : "d", sx: gauss(), sy: gauss(), delay: R() * 0.55 });
  }
  function layout() {
    const w = cv.clientWidth || 400, dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.round(w * dpr); cv.height = Math.round(w * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const L = 34, Tp = 34, g = w - L - 8, c = g / K;
    geo = { w, L, Tp, c };
    dots = base.map(d => {
      const inner = c * 0.86, sp = inner / d.side, off = (c - inner) / 2 + sp / 2;
      const r = Math.floor(d.k / d.side), q = d.k % d.side;
      return { ...d, tx: L + d.j * c + off + q * sp, ty: Tp + d.i * c + off + r * sp,
        fx: w * 0.55 + d.sx * w * 0.16, fy: w * 0.5 + d.sy * w * 0.16, size: Math.max(1.2, Math.min(2.6, sp * 0.8)) };
    });
  }
  function draw(p) {
    const { w, L, Tp, c } = geo;
    ctx.clearRect(0, 0, w, w);
    const tint = clamp((p - 1.05) / 0.5, 0, 1);
    // cells
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      ctx.fillStyle = C.mid;
      ctx.fillRect(L + j * c + 0.75, Tp + i * c + 0.75, c - 1.5, c - 1.5);
      if (tint > 0 && Wt[i][j] > 0) {
        ctx.globalAlpha = tint * 0.6;
        ctx.fillStyle = ramp("pos", Wt[i][j] * 0.75);
        ctx.fillRect(L + j * c + 0.75, Tp + i * c + 0.75, c - 1.5, c - 1.5);
        ctx.globalAlpha = 1;
      }
    }
    // ticks
    ctx.fillStyle = C.muted; ctx.font = `10px ${getComputedStyle(document.body).getPropertyValue("--mono") || "monospace"}`;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let k = 0; k < K; k++) ctx.fillText(String(k), L - 5, Tp + k * c + c / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    for (let k = 0; k < K; k++) ctx.fillText(String(k), L + k * c + c / 2, Tp - 6);
    ctx.fillStyle = C["ink-2"]; ctx.font = `600 11px Archivo, system-ui, sans-serif`;
    ctx.fillText(T("CHP affinity  j  →", "CHP yakınlığı  j  →"), L + K * c / 2, 11);
    ctx.save(); ctx.translate(10, Tp + K * c / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(T("←  i  AKP affinity", "←  i  AKP yakınlığı"), 0, 0); ctx.restore();
    // dots
    const col = { x: C["camp-x"], y: C["camp-y"], d: C.diag };
    for (const camp of ["d", "x", "y"]) {
      ctx.fillStyle = col[camp];
      for (const d of dots) {
        if (d.camp !== camp) continue;
        const t = ease(clamp((p - d.delay) / 0.45, 0, 1));
        const x = lerp(d.fx, d.tx, t), y = lerp(d.fy, d.ty, t);
        ctx.fillRect(x - d.size / 2, y - d.size / 2, d.size, d.size);
      }
    }
    if (tint > 0) {
      ctx.globalAlpha = tint;
      ctx.font = `700 12px Archivo, system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const lab = (txt, i, j, dx, dy) => {
        const x = L + j * c + c / 2 + dx, y = Tp + i * c + c / 2 + dy;
        ctx.lineWidth = 4; ctx.strokeStyle = C.sheet; ctx.strokeText(txt, x, y);
        ctx.fillStyle = C.ink; ctx.fillText(txt, x, y);
      };
      lab("311", 0, 10, 0, 0);
      lab("259", 10, 0, 0, 0);
      ctx.textAlign = "right";
      ctx.font = `500 12px ${"IBM Plex Mono"}, monospace`;
      const sx = L + K * c - 4, sy = Tp + K * c - 12;
      ctx.lineWidth = 4; ctx.strokeStyle = C.sheet;
      const s = "S_pol = 0.453";
      ctx.strokeText(s, sx, sy); ctx.fillStyle = C.ink; ctx.fillText(s, sx, sy);
      ctx.globalAlpha = 1;
    }
  }
  function frame(now) {
    prog = (now - t0) / 1000 / 2.2;
    draw(Math.min(prog, 1.6));
    if (prog < 1.6) raf = requestAnimationFrame(frame);
  }
  function play() {
    cancelAnimationFrame(raf);
    if (REDUCED) { prog = 99; draw(1.6); return; }
    t0 = performance.now(); raf = requestAnimationFrame(frame);
  }
  layout();
  draw(Math.min(prog, 1.6));
  new ResizeObserver(() => { layout(); draw(Math.min(prog, 1.6)); }).observe(cv);
  $("#heroReplay").addEventListener("click", play);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => draw(Math.min(prog, 1.6)));
  onTheme(() => draw(Math.min(prog, 1.6)));
  onLang(() => draw(Math.min(prog, 1.6)));
  return { play };
}
