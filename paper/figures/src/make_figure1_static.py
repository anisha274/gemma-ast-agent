W, H = 1600, 640
F = "Liberation Serif, Times New Roman, serif"
BLUE = ("#e9f1fb", "#2b6cb0"); ORNG = ("#fdf1e5", "#c05621"); GREY = ("#f5f5f5", "#4a5568")
out = []
def text(x, y, s, size=25, color="#1a202c", anchor="middle", bold=False, italic=False):
    out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"'
               + (' font-weight="bold"' if bold else '') + (' font-style="italic"' if italic else '') + f'>{s}</text>')
def box(x, y, w, h, col, title, lines):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{col[0]}" stroke="{col[1]}" stroke-width="2.5"/>')
    text(x + w/2, y + 42, title, 28, col[1], bold=True)
    for i, ln in enumerate(lines):
        text(x + w/2, y + 86 + i * 32, ln)
def arrow(pts, color="#2d3748", dashed=False, width=2.8):
    d = "M " + " L ".join(f"{a},{b}" for a, b in pts)
    m = "aO" if color == ORNG[1] else "aG"
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"' + (' stroke-dasharray="10 7"' if dashed else '') + f' marker-end="url(#{m})"/>')

Y, BH, BW, GAP, X0 = 110, 230, 222, 40, 20
xs = [X0 + i * (BW + GAP) for i in range(6)]
spec = [(GREY, "Input", ["Issue (failing", "tests)", "Repository", "Test command"]),
        (BLUE, "① AST index", ["Functions,", "methods, class", "outlines", "+ call graph"]),
        (BLUE, "② Retrieval", ["BM25 + issue /", "traceback boosts", "+ call-graph", "≤ 6,000 tokens"]),
        (GREY, "③ Gemma 4", ["E2B or E4B", "greedy decoding", "→ SEARCH/", "REPLACE edits"]),
        (GREY, "④ Apply edits", ["Atomic,", "syntax-checked;", "tests are", "read-only"]),
        (GREY, "⑤ Run tests", ["Library’s own", "test suite", "Pass →", "output patch"])]
for x, (col, t, l) in zip(xs, spec):
    box(x, Y, BW, BH, col, t, l)
cy = Y + BH / 2
for i in range(5):
    arrow([(xs[i] + BW + 2, cy), (xs[i + 1] - 7, cy)])
# rejected-edit loop (top): Apply -> Gemma
a, g = xs[4] + BW / 2, xs[3] + BW / 2
arrow([(a, Y - 2), (a, Y - 48), (g, Y - 48), (g, Y - 9)], ORNG[1], dashed=True)
text((a + g) / 2, Y - 60, "edit rejected → error message", 23, ORNG[1], italic=True)
# self-correction loop (bottom): Run tests -> box -> Gemma
t = xs[5] + BW / 2
LY = Y + BH + 115
bx, bh = xs[3] - 10, 110
bw = (t - 40) - bx
arrow([(t, Y + BH + 2), (t, LY), (bx + bw + 7, LY)], ORNG[1], width=3)
out.append(f'<rect x="{bx}" y="{LY - bh/2}" width="{bw}" height="{bh}" rx="12" fill="{ORNG[0]}" stroke="{ORNG[1]}" stroke-width="2.5"/>')
text(bx + bw / 2, LY - 18, "Self-correction: tests fail and attempt &lt; 3", 25, ORNG[1], bold=True)
text(bx + bw / 2, LY + 20, "reset; send diff + test output + unseen traceback code", 23)
arrow([(g, LY - bh / 2 - 1), (g, Y + BH + 9)], ORNG[1], width=3)
text(t + 12, Y + BH + 50, "fail", 22, ORNG[1], anchor="start", italic=True)
# legend
LGY = 600
def sw(x, col, s):
    out.append(f'<rect x="{x}" y="{LGY - 20}" width="30" height="22" rx="4" fill="{col[0]}" stroke="{col[1]}" stroke-width="2.2"/>')
    text(x + 40, LGY, s, 23, anchor="start")
sw(20, BLUE, "AST-guided retrieval"); sw(380, ORNG, "Test-driven self-correction"); sw(820, GREY, "Standard components")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{F}">
<defs>
<marker id="aG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2d3748"/></marker>
<marker id="aO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{ORNG[1]}"/></marker>
</defs><rect width="{W}" height="{H}" fill="#fff"/>
{chr(10).join(out)}</svg>'''
open("figure1_workflow.svg", "w").write(svg)
open("fig.html", "w").write(f'<html><body style="margin:0">{svg}</body></html>')
