import sys, pathlib
src = pathlib.Path(sys.argv[1]); out_repo = pathlib.Path(sys.argv[2]); out_frag = pathlib.Path(sys.argv[3])
head = (src/'head.html').read_text(); css = (src/'style.css').read_text(); body = (src/'body.html').read_text()
js = "\n".join((src/f).read_text() for f in ['data.js','core.js','chapters.js'])
frag = f"{head}<style>\n{css}</style>\n{body}\n<script>\n{js}</script>\n"
standalone_reset = ":root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}"
full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        f"<style>{standalone_reset}</style>\n{head}<style>\n{css}</style>\n</head>\n<body>\n{body}\n<script>\n{js}</script>\n</body>\n</html>\n")
out_repo.parent.mkdir(parents=True, exist_ok=True)
out_repo.write_text(full); out_frag.write_text(frag)
print(len(full), "bytes ->", out_repo); print(len(frag), "bytes ->", out_frag)
