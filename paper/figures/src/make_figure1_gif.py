import json
W, H = 1600, 640
F = "Liberation Serif, Times New Roman, serif"
BLUE = ("#e9f1fb", "#2b6cb0"); ORNG = ("#fdf1e5", "#c05621"); GREY = ("#f5f5f5", "#4a5568")
RED = ("#fde8e8", "#c53030"); GRN = ("#e6f6ec", "#2f855a")
Y, BH, BW, GAP, X0 = 110, 230, 222, 40, 20
xs = [X0 + i * (BW + GAP) for i in range(6)]
SPEC = [(GREY, "Input", ["Issue (failing", "tests)", "Repository", "Test command"]),
        (BLUE, "① AST index", ["Functions,", "methods, class", "outlines", "+ call graph"]),
        (BLUE, "② Retrieval", ["BM25 + issue /", "traceback boosts", "+ call-graph", "≤ 6,000 tokens"]),
        (GREY, "③ Gemma 4", ["E2B or E4B", "greedy decoding", "→ SEARCH/", "REPLACE edits"]),
        (GREY, "④ Apply edits", ["Atomic,", "syntax-checked;", "tests are", "read-only"]),
        (GREY, "⑤ Run tests", ["Library’s own", "test suite", "Pass →", "output patch"])]

def build(active=None, tests_col=None, loop_on=False, caption="", attempt="", final=False):
    """active: set of box indices highlighted (None = all normal)."""
    out = []
    def text(x, y, s, size=25, color="#1a202c", anchor="middle", bold=False, italic=False, op=1):
        out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" opacity="{op}"'
                   + (' font-weight="bold"' if bold else '') + (' font-style="italic"' if italic else '') + f'>{s}</text>')
    def arrow(pts, color="#2d3748", dashed=False, width=2.8, op=1):
        d = "M " + " L ".join(f"{a},{b}" for a, b in pts)
        m = {"#2d3748": "aG", ORNG[1]: "aO", RED[1]: "aR", GRN[1]: "aN"}[color]
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{op}"' + (' stroke-dasharray="10 7"' if dashed else '') + f' marker-end="url(#{m})"/>')
    dim = lambda i: 1.0 if (active is None or i in active) else 0.28
    for i, (x, (col, t, lines)) in enumerate(zip(xs, SPEC)):
        if i == 5 and tests_col:
            col = tests_col
            lines = ["Library\u2019s own", "test suite", "FAIL:", "1 test fails"] if tests_col == RED else ["Library\u2019s own", "test suite", "PASS \u2192", "output patch"]
        on = active is not None and i in active
        op = dim(i)
        flt = ' filter="url(#glow)"' if on else ''
        out.append(f'<g opacity="{op}"><rect x="{x}" y="{Y}" width="{BW}" height="{BH}" rx="12" fill="{col[0]}" stroke="{col[1]}" stroke-width="{4.5 if on else 2.5}"{flt}/>')
        text(x + BW/2, Y + 42, t, 28, col[1], bold=True)
        for k, ln in enumerate(lines):
            text(x + BW/2, Y + 86 + k * 32, ln)
        out.append('</g>')
    cy = Y + BH / 2
    for i in range(5):
        op = 1.0 if active is None or (i + 1) in (active or set()) else 0.28
        arrow([(xs[i] + BW + 2, cy), (xs[i + 1] - 7, cy)], op=op)
    a, g = xs[4] + BW / 2, xs[3] + BW / 2
    top_op = 1.0 if final else 0.28
    arrow([(a, Y - 2), (a, Y - 48), (g, Y - 48), (g, Y - 9)], ORNG[1], dashed=True, op=top_op)
    text((a + g) / 2, Y - 60, "edit rejected → error message", 23, ORNG[1], italic=True, op=top_op)
    t = xs[5] + BW / 2
    LY = Y + BH + 115; bx, bh = xs[3] - 10, 110; bw = (t - 40) - bx
    lop = 1.0 if (loop_on or final) else 0.28
    lw = 5 if loop_on else 3
    arrow([(t, Y + BH + 2), (t, LY), (bx + bw + 7, LY)], ORNG[1], width=lw, op=lop)
    flt = ' filter="url(#glow)"' if loop_on else ''
    out.append(f'<g opacity="{lop}"><rect x="{bx}" y="{LY - bh/2}" width="{bw}" height="{bh}" rx="12" fill="{ORNG[0]}" stroke="{ORNG[1]}" stroke-width="{4.5 if loop_on else 2.5}"{flt}/>')
    text(bx + bw / 2, LY - 18, "Self-correction: tests fail and attempt &lt; 3", 25, ORNG[1], bold=True)
    text(bx + bw / 2, LY + 20, "reset; send diff + test output + unseen traceback code", 23)
    out.append('</g>')
    arrow([(g, LY - bh / 2 - 1), (g, Y + BH + 9)], ORNG[1], width=lw, op=lop)
    text(t + 12, Y + BH + 50, "fail", 22, ORNG[1], anchor="start", italic=True, op=lop)
    if attempt:
        out.append(f'<rect x="20" y="22" width="190" height="44" rx="22" fill="#2d3748"/>')
        text(115, 52, attempt, 24, "#ffffff", bold=True)
    if final:
        LGY = 600
        for x, col, s in [(20, BLUE, "AST-guided retrieval"), (380, ORNG, "Test-driven self-correction"), (820, GREY, "Standard components")]:
            out.append(f'<rect x="{x}" y="{LGY - 20}" width="30" height="22" rx="4" fill="{col[0]}" stroke="{col[1]}" stroke-width="2.2"/>')
            text(x + 40, LGY, s, 23, anchor="start")
        text(W - 20, LGY, "Retrieval finds the bug; test feedback fixes it.", 25, "#2b6cb0", anchor="end", italic=True, bold=True)
    elif caption:
        text(W / 2, 604, caption, 26, "#1a202c")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{F}">
<defs>
<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="0" stdDeviation="7" flood-color="#f6ad55" flood-opacity="0.9"/></filter>
<marker id="aG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2d3748"/></marker>
<marker id="aO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{ORNG[1]}"/></marker>
<marker id="aR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{RED[1]}"/></marker>
<marker id="aN" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{GRN[1]}"/></marker>
</defs><rect width="{W}" height="{H}" fill="#fff"/>
{chr(10).join(out)}</svg>'''

frames = [
  (dict(active={0}, caption="Input: a bug report listing the failing tests, the repository, and its test command"), 1900),
  (dict(active={0, 1}, caption="① Split the code into functions, methods and class outlines, linked by a call graph"), 1900),
  (dict(active={1, 2}, caption="② Pick the most relevant functions for the bug report, within 6,000 tokens", attempt="Attempt 1"), 1900),
  (dict(active={2, 3}, caption="③ Gemma 4 reads the code and writes SEARCH/REPLACE edits", attempt="Attempt 1"), 1700),
  (dict(active={3, 4}, caption="④ Apply the edits safely: all or nothing, syntax-checked, tests untouched", attempt="Attempt 1"), 1700),
  (dict(active={4, 5}, tests_col=RED, caption="⑤ Run the tests: attempt 1 fails", attempt="Attempt 1"), 1700),
  (dict(active={5}, tests_col=RED, loop_on=True, caption="Self-correction: reset the code, show the model what failed and why", attempt="Attempt 1"), 2300),
  (dict(active={3}, loop_on=True, caption="③ Gemma 4 writes a new patch using the test feedback", attempt="Attempt 2"), 1800),
  (dict(active={3, 4}, caption="④ Apply the new edits", attempt="Attempt 2"), 1300),
  (dict(active={4, 5}, tests_col=GRN, caption="⑤ The tests pass → output the patch", attempt="Attempt 2"), 2000),
  (dict(final=True), 4500),
]
svgs = [build(**kw) for kw, _ in frames]
json.dump({"svgs": svgs, "durations": [d for _, d in frames]}, open("frames.json", "w"))
open("final_frame.svg", "w").write(svgs[-1])
print(len(svgs), "frames,", sum(d for _, d in frames) / 1000, "s")
