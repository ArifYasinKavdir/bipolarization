/* ============================================================
   Data
   TOY  - invented 20-respondent survey on a 0-4 scale.
   SIM  - simulated AKP (rows, i) x CHP (columns, j) count table, n = 2,218.
          Reconstructed so that W o P matches Figure 2(c) of the paper to
          three decimals and the shares, scores and margins match the
          published values. Not the TGSS microdata.
   PARTIES, T3, T4, T6 - published values from Tables 2, 3, 4 and 6.
   ============================================================ */
const TOY = [
  [3, 0, 1, 1, 4],
  [0, 1, 0, 1, 0],
  [1, 0, 2, 0, 0],
  [1, 1, 0, 0, 0],
  [3, 0, 0, 0, 1],
];
const SIM = [
  [309, 28, 7, 19, 21, 85, 49, 68, 98, 36, 311],
  [22, 10, 1, 1, 1, 5, 10, 3, 11, 11, 7],
  [1, 1, 10, 5, 2, 9, 7, 12, 9, 2, 5],
  [13, 1, 10, 10, 15, 20, 12, 13, 3, 1, 6],
  [7, 1, 2, 15, 12, 12, 16, 1, 4, 0, 0],
  [58, 5, 1, 20, 12, 66, 13, 7, 7, 0, 22],
  [22, 1, 1, 12, 5, 13, 10, 1, 1, 2, 0],
  [48, 3, 12, 8, 1, 7, 1, 8, 1, 0, 0],
  [51, 2, 9, 8, 4, 9, 1, 1, 8, 1, 7],
  [37, 3, 1, 1, 1, 2, 2, 2, 0, 4, 1],
  [259, 10, 7, 11, 9, 22, 6, 12, 7, 1, 1],
];
/* Table 2 */
const PARTIES = [
  { a: "CHP", en: "Republican People's Party", tr: "Cumhuriyet Halk Partisi", n: 2268, mean: 4.09, sd: 3.85, at0: 37.3, ge8: 25.4, spol: 0.304, scon: -0.382, ss: 5.00 },
  { a: "AKP", en: "Justice and Development Party", tr: "Adalet ve Kalkınma Partisi", n: 2272, mean: 3.44, sd: 3.90, at0: 46.5, ge8: 22.6, spol: 0.264, scon: -0.455, ss: 4.96 },
  { a: "MHP", en: "Nationalist Movement Party", tr: "Milliyetçi Hareket Partisi", n: 2233, mean: 2.96, sd: 3.58, at0: 49.2, ge8: 16.4, spol: 0.221, scon: -0.491, ss: 4.31 },
  { a: "YRP", en: "New Welfare Party", tr: "Yeniden Refah Partisi", n: 2159, mean: 2.01, sd: 2.97, at0: 59.0, ge8: 8.6, spol: 0.157, scon: -0.582, ss: 3.20 },
  { a: "DEM", en: "Peoples' Equality and Democracy Party", tr: "Halkların Eşitlik ve Demokrasi Partisi", n: 2259, mean: 1.06, sd: 2.57, at0: 79.6, ge8: 6.2, spol: 0.155, scon: -0.664, ss: 4.40 },
  { a: "ZP", en: "Victory Party", tr: "Zafer Partisi", n: 2125, mean: 1.74, sd: 2.88, at0: 63.7, ge8: 7.4, spol: 0.153, scon: -0.609, ss: 3.22 },
  { a: "TİP", en: "Workers' Party of Turkey", tr: "Türkiye İşçi Partisi", n: 2125, mean: 1.44, sd: 2.60, at0: 68.3, ge8: 5.0, spol: 0.146, scon: -0.630, ss: 2.98 },
  { a: "İYİ", en: "Good Party", tr: "İYİ Parti", n: 2204, mean: 1.45, sd: 2.32, at0: 62.5, ge8: 2.5, spol: 0.126, scon: -0.631, ss: 1.96 },
  { a: "MP", en: "Homeland Party", tr: "Memleket Partisi", n: 2113, mean: 1.31, sd: 2.34, at0: 67.4, ge8: 3.2, spol: 0.125, scon: -0.649, ss: 2.27 },
  { a: "SP", en: "Felicity Party", tr: "Saadet Partisi", n: 2163, mean: 1.18, sd: 2.19, at0: 68.8, ge8: 2.6, spol: 0.122, scon: -0.661, ss: 2.05 },
  { a: "HÜDAPAR", en: "Free Cause Party", tr: "Hür Dava Partisi", n: 2144, mean: 0.61, sd: 1.69, at0: 83.2, ge8: 1.2, spol: 0.120, scon: -0.707, ss: 2.24 },
  { a: "DEVA", en: "Democracy and Progress Party", tr: "Demokrasi ve Atılım Partisi", n: 2141, mean: 0.80, sd: 1.80, at0: 76.8, ge8: 1.2, spol: 0.113, scon: -0.698, ss: 1.62 },
  { a: "GP", en: "Future Party", tr: "Gelecek Partisi", n: 2067, mean: 0.82, sd: 1.80, at0: 75.5, ge8: 1.2, spol: 0.111, scon: -0.698, ss: 1.48 },
];
/* Table 3: rows/cols in this order; lower triangle S_pol, upper S_con */
const T3_ORDER = ["AKP", "CHP", "DEVA", "GP", "HÜDAPAR", "DEM", "İYİ", "MP", "MHP", "SP", "TİP", "YRP", "ZP"];
const T3_TEXT = `
AKP — −.20 −.53 −.53 −.56 −.49 −.46 −.48 −.36 −.50 −.45 −.44 −.44
CHP .45 — −.46 −.46 −.45 −.43 −.40 −.42 −.24 −.41 −.41 −.33 −.38
DEVA .25 .28 — −.82 −.83 −.78 −.73 −.75 −.57 −.77 −.74 −.68 −.71
GP .24 .27 .03 — −.82 −.78 −.74 −.76 −.58 −.77 −.74 −.68 −.71
HÜDAPAR .23 .31 .04 .04 — −.81 −.74 −.76 −.59 −.78 −.75 −.70 −.71
DEM .31 .29 .08 .08 .08 — −.69 −.71 −.52 −.73 −.72 −.64 −.67
İYİ .25 .26 .06 .06 .08 .12 — −.69 −.51 −.69 −.66 −.61 −.65
MP .26 .26 .06 .06 .08 .12 .07 — −.53 −.71 −.68 −.62 −.67
MHP .15 .39 .20 .20 .19 .26 .19 .20 — −.54 −.49 −.47 −.49
SP .23 .29 .05 .05 .06 .10 .08 .08 .20 — −.69 −.65 −.67
TİP .30 .24 .09 .08 .09 .09 .11 .10 .26 .10 — −.60 −.64
YRP .21 .33 .11 .11 .11 .16 .13 .13 .18 .10 .16 — −.58
ZP .28 .26 .10 .10 .12 .16 .10 .08 .22 .12 .13 .16 —`;
const T3 = T3_TEXT.trim().split("\n").map(line => line.trim().split(/\s+/).slice(1).map(s => (s === "—" ? 0 : parseFloat(s.replace("−", "-")))));
/* Table 4: ten most and five least polarized dyads */
const T4 = [
  ["AKP × CHP", 2218, 35.2, 44.6, 0.453, 0.436, 0.469, 5.68, 5.66, 0.02, -0.29, 0.35],
  ["CHP × MHP", 2191, 45.1, 30.6, 0.393, 0.377, 0.409, 5.45, 4.79, 0.66, 0.32, 1.01],
  ["CHP × YRP", 2129, 48.1, 21.9, 0.326, 0.311, 0.341, 5.22, 3.39, 1.83, 1.49, 2.15],
  ["CHP × HÜDAPAR", 2113, 57.3, 7.2, 0.315, 0.299, 0.330, 5.21, 2.29, 2.93, 2.45, 3.36],
  ["AKP × DEM", 2198, 47.2, 12.5, 0.308, 0.292, 0.324, 5.25, 4.87, 0.38, -0.12, 0.89],
  ["AKP × TİP", 2089, 43.9, 19.6, 0.299, 0.284, 0.314, 5.26, 3.46, 1.80, 1.44, 2.17],
  ["CHP × DEM", 2201, 53.7, 8.2, 0.293, 0.277, 0.308, 4.91, 3.59, 1.32, 0.71, 1.85],
  ["CHP × SP", 2138, 52.5, 11.6, 0.292, 0.277, 0.306, 5.06, 2.25, 2.81, 2.43, 3.18],
  ["AKP × ZP", 2097, 39.8, 21.6, 0.285, 0.269, 0.301, 5.30, 3.41, 1.89, 1.50, 2.27],
  ["CHP × DEVA", 2110, 54.8, 5.2, 0.277, 0.263, 0.292, 4.91, 1.57, 3.34, 2.89, 3.72],
  ["DEVA × GP", 2032, 9.5, 10.6, 0.029, 0.024, 0.033, 1.48, 1.36, 0.12, -0.22, 0.47],
  ["DEVA × HÜDAPAR", 2075, 15.5, 8.2, 0.044, 0.039, 0.049, 1.59, 2.32, -0.73, -1.17, -0.29],
  ["GP × HÜDAPAR", 2012, 16.7, 8.6, 0.045, 0.039, 0.051, 1.52, 2.24, -0.72, -1.17, -0.29],
  ["GP × SP", 2035, 10.5, 17.7, 0.051, 0.045, 0.057, 1.49, 1.98, -0.49, -0.84, -0.14],
  ["DEVA × SP", 2099, 9.2, 18.2, 0.052, 0.046, 0.058, 1.57, 2.03, -0.46, -0.79, -0.09],
];
/* Table 6: [label en, label tr, opts, max|m|, Spol, lo, hi, mean, min, max, rho] */
const T6 = [
  ["Power, p = 1, q = 1 (default)", "Kuvvet, p = 1, q = 1 (varsayılan)", { kernel: "power", p: 1, q: 1 }, 1.0, 0.453, 0.436, 0.469, 0.163, 0.029, 0.453, 1.0],
  ["Power, p = 0.5, q = 1", "Kuvvet, p = 0.5, q = 1", { kernel: "power", p: 0.5, q: 1 }, 1.0, 0.512, 0.496, 0.528, 0.201, 0.048, 0.512, 0.9954],
  ["Power, p = 2, q = 1", "Kuvvet, p = 2, q = 1", { kernel: "power", p: 2, q: 1 }, 1.0, 0.386, 0.369, 0.403, 0.124, 0.014, 0.386, 0.9968],
  ["Power, p = 1, q = 0", "Kuvvet, p = 1, q = 0", { kernel: "power", p: 1, q: 0 }, 1.0, 0.534, 0.518, 0.550, 0.222, 0.052, 0.534, 0.9949],
  ["Power, p = 1, q = 2", "Kuvvet, p = 1, q = 2", { kernel: "power", p: 1, q: 2 }, 1.0, 0.404, 0.388, 0.421, 0.133, 0.019, 0.404, 0.9989],
  ["Gaussian, σ = 0.25, q = 1", "Gauss, σ = 0.25, q = 1", { kernel: "gaussian", sigma: 0.25, q: 1 }, 0.9997, 0.532, 0.516, 0.549, 0.206, 0.040, 0.532, 0.9970],
  ["Gaussian, σ = 0.5, q = 1", "Gauss, σ = 0.5, q = 1", { kernel: "gaussian", sigma: 0.5, q: 1 }, 0.8647, 0.443, 0.426, 0.459, 0.154, 0.022, 0.443, 0.9990],
  ["Gaussian, σ = 1, q = 1", "Gauss, σ = 1, q = 1", { kernel: "gaussian", sigma: 1, q: 1 }, 0.3935, 0.401, 0.384, 0.417, 0.132, 0.016, 0.401, 0.9976],
];
