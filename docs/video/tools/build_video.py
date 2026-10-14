import json, pathlib, sys
V = pathlib.Path(sys.argv[1]); SRC = pathlib.Path(sys.argv[2])
OUT = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else V / "video.html"
timing = json.load(open(V / "timing.json"))
tok = [{"t": t["t"], "para": t["para"], "s": round(t["s"], 3), "e": round(t["e"], 3)} for t in timing]
js = "\n".join([(SRC / "data.js").read_text(), (SRC / "core.js").read_text(), "const TIMING = " + json.dumps(tok, ensure_ascii=False) + ";", (V / "video.js").read_text()])
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Bipolarization video</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..100,400..800&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400..600;1,8..60,400&display=block">
<script>
window.MathJax = {{ tex: {{ inlineMath: [] }}, svg: {{ fontCache: "local" }}, options: {{ enableMenu: false, enableAssistiveMml: false }},
  startup: {{ typeset: false, ready() {{ MathJax.startup.defaultReady(); MathJax.startup.promise.then(function () {{ window.__mjReady = true; }}); }} }} }};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>
<style>{(V / "video.css").read_text()}</style></head>
<body><div id="stage"></div>
<script>
{js}
</script></body></html>"""
OUT.write_text(html)
print(OUT, len(html), "bytes")
