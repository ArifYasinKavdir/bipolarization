import json, re, sys, difflib
script = open(sys.argv[1]).read()
words = json.load(open(sys.argv[2]))
out = sys.argv[3]
NUM = {"zero": "0", "ten": "10", "eleven": "11", "2024": "2024"}
def norm(t):
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9']", "", t)
    return NUM.get(t, t)
# script tokens with paragraph index
paras = [p.strip() for p in script.strip().split("\n\n")]
toks = []
for pi, p in enumerate(paras):
    p = re.sub(r"\[[^\]]+\]", "", p)
    for raw in re.findall(r"\S+", p.replace("…", " … ").replace("-", " - ")):
        if raw in ("…", "-"):
            continue
        toks.append({"t": raw, "n": norm(raw), "para": pi})
# transcript tokens (split hyphen pieces like "per -person")
tw = []
for w in words:
    for piece in w["w"].replace("-", " ").split():
        n = norm(piece)
        if n:
            tw.append({"n": n, "s": w["s"], "e": w["e"]})
sm = difflib.SequenceMatcher(a=[t["n"] for t in toks], b=[w["n"] for w in tw], autojunk=False)
for a, b, size in sm.get_matching_blocks():
    for k in range(size):
        toks[a + k]["s"] = tw[b + k]["s"]; toks[a + k]["e"] = tw[b + k]["e"]
# interpolate unmatched
known = [i for i, t in enumerate(toks) if "s" in t]
for i, t in enumerate(toks):
    if "s" in t: continue
    prev = max([k for k in known if k < i], default=None); nxt = min([k for k in known if k > i], default=None)
    if prev is None: t["s"] = toks[nxt]["s"]; t["e"] = toks[nxt]["s"]
    elif nxt is None: t["s"] = toks[prev]["e"]; t["e"] = toks[prev]["e"]
    else:
        span = nxt - prev; f0 = (i - prev) / span
        t["s"] = toks[prev]["e"] + (toks[nxt]["s"] - toks[prev]["e"]) * f0
        t["e"] = t["s"] + 0.25
    t["interp"] = True
matched = sum(1 for t in toks if not t.get("interp"))
print(f"script tokens {len(toks)}, matched {matched} ({100*matched/len(toks):.1f}%)")
for pi in range(len(paras)):
    ts = [t for t in toks if t["para"] == pi]
    print(f"para {pi}: {ts[0]['s']:7.2f} – {ts[-1]['e']:7.2f}  {' '.join(t['t'] for t in ts[:6])} …")
json.dump(toks, open(out, "w"))
print("unmatched:", [t["t"] for t in toks if t.get("interp")])
