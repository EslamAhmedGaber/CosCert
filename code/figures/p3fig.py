"""Paper 3 figure system.

* Typesetting: every string is typeset by XeLaTeX through matplotlib's PGF backend with the SAME preamble as the
  manuscript (work/manuscript/p3fonts.tex: Latin Modern text with optical sizes, Latin Modern Math).
* Physical size: figures are drawn at their printed width (elsarticle 3p text width, 468 pt), so the pt sizes below
  are the printed sizes. Ladder (role-mapped): 9 pt labels/titles, 8 pt legends/annotations, 7 pt tick labels.
* Colour = procedure, fixed across the paper (Okabe-Ito, CVD-safe); lightness within a hue = the procedure's own
  tuning parameter (m, B) or the memory parameter H.  Grey = references (nominal level, H = 1/2 values).
* save() writes <name>.pdf (vector, fonts embedded by XeLaTeX), a 200-dpi PNG preview and <name>.meta.json (inputs).
"""
import os, json, time, hashlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex

WORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # work/ (code/figures/p3fig.py)
TEXBIN = r"C:\texlive\2025\bin\windows"
FIGDIR = os.path.join(WORK, "figures")
FONTS_TEX = os.path.join(WORK, "manuscript", "p3fonts.tex")

MM = 1 / 25.4
TEXTWIDTH = 468.0 / 72.27       # inches: \textwidth of elsarticle [final,3p,11pt] (measured with XeLaTeX, 468 pt = 164.6 mm)
SIZES = (9, 8, 7)               # base (labels, titles) / legend, annotation / tick labels   [pt]

COL = {
    "SN": "#0072B2",      # self-normalised / Bartlett b = 1 (T_H)          -- focal procedure
    "HI": "#D55E00",      # Hualde-Iacone fixed-m periodogram
    "EWC": "#009E73",     # equal-weighted cosine (LLSW)
    "MP": "#CC79A7",      # McElroy-Politis published fit
    "H05": "#E69F00",     # published H = 1/2 values (Lobato, Kiefer-Vogelsang, Abadir-Paruolo)
    "KPSS": "#56B4E9",
    "REF": "#7A7A7A",     # nominal level, reference lines
    "INK": "#1A1A1A",     # text / certified marks
}


def tex_environment():
    """XeLaTeX from the user's TeX Live 2025 (read-only); every TeX cache/config redirected into the project."""
    if TEXBIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = TEXBIN + os.pathsep + os.environ.get("PATH", "")
    tv = os.path.join(WORK, ".texmf_var")
    os.environ.update({"TEXMFVAR": tv, "TEXMFCACHE": tv, "TEXMFCONFIG": os.path.join(tv, "config"),
                       "TEXMFHOME": os.path.join(tv, "home")})


def setup():
    tex_environment()
    base, small, tick = SIZES
    mpl.rcParams.update({
        "pgf.texsystem": "xelatex", "pgf.rcfonts": False,
        # matplotlib's PGF preamble forces \displaystyle on all math; reset it so figure math matches inline text math
        "pgf.preamble": open(FONTS_TEX, encoding="utf-8").read() + "\n\\everymath{}\n",
        "font.family": "serif", "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "legend.fontsize": small, "xtick.labelsize": tick, "ytick.labelsize": tick, "figure.titlesize": base,
        "axes.titlelocation": "left", "axes.titlepad": 5, "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
        "axes.labelpad": 3, "axes.grid": False, "axes.axisbelow": True,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45, "ytick.minor.width": 0.45, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.7, "ytick.minor.size": 1.7, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "lines.linewidth": 1.25, "lines.markersize": 3.6, "lines.markeredgewidth": 0.8, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3, "legend.columnspacing": 1.0,
        "grid.color": "#E3E3E3", "grid.linewidth": 0.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "standard", "savefig.pad_inches": 0.0,
        "axes.unicode_minus": True, "text.usetex": False,
    })


def tint(color, t):
    """Mix `color` with white: t = 1 -> colour itself, t -> 0 -> white."""
    c = np.array(to_rgb(color)); return to_hex(1.0 - t * (1.0 - c))


def shade(color, t):
    """Mix `color` with black: t = 0 -> colour itself, t = 1 -> black."""
    c = np.array(to_rgb(color)); return to_hex(c * (1.0 - t))


def ramp(color, n, lo=0.38, dark=0.35):
    """n colours within one hue, light -> dark (light tints, the pure hue, then darker shades)."""
    if n == 1:
        return [color]
    ts = np.linspace(0.0, 1.0, n); out = []
    for t in ts:
        out.append(tint(color, lo + (1 - lo) * t / 0.6) if t <= 0.6 else shade(color, dark * (t - 0.6) / 0.4))
    return out


def panel_tag(ax, letter, x=-0.02, y=1.0, fig=None):
    """Bold (a)-style tag at the top-left corner of the axes' outer box (figure coordinates offset in points)."""
    ax.annotate(rf"\textbf{{({letter})}}", xy=(0, 1), xycoords="axes fraction", xytext=(-24 + 72 * x, 6 + 72 * (y - 1)),
                textcoords="offset points", ha="left", va="bottom", fontsize=SIZES[0], annotation_clip=False)


def log_ticks(ax, axis="y", ticks=None, fmt=None):
    from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    if ticks is not None:
        a.set_major_locator(FixedLocator(ticks))
        a.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: (fmt or thousands)(v)))
    a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 1.0, numticks=100))
    a.set_minor_formatter(NullFormatter())


def thousands(v):
    if v >= 10000:
        s = f"{v:,.0f}".replace(",", r"\,"); return s
    if v >= 1:
        return f"{v:g}"
    return f"{v:g}"


def pct(v, digits=0):
    return f"{100 * v:.{digits}f}"


def direct_labels(ax, x, ys, labels, colors, dx_pt=3.0, min_gap_pt=None, fontsize=None, ha="left", va="center"):
    """Labels at (x, y_i) shifted right by dx_pt; y positions de-overlapped in display space (points)."""
    fs = fontsize or SIZES[1]; gap = min_gap_pt or 1.15 * fs
    tr = ax.transData; inv = ax.transData.inverted(); fig = ax.figure
    dpi = fig.dpi
    pts = np.array([tr.transform((x, y))[1] for y in ys]) * 72.0 / dpi
    order = np.argsort(pts); placed = pts.copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(order)):
            a, b = order[i - 1], order[i]
            if placed[b] - placed[a] < gap:
                d = (gap - (placed[b] - placed[a])) / 2; placed[a] -= d; placed[b] += d; moved = True
        if not moved:
            break
    texts = []
    for i, (lab, c) in enumerate(zip(labels, colors)):
        yd = placed[i] * dpi / 72.0; xd = tr.transform((x, ys[i]))[0]
        xy_data = inv.transform((xd, yd))
        texts.append(ax.annotate(lab, xy=(x, xy_data[1]), xytext=(dx_pt, 0), textcoords="offset points", ha=ha, va=va,
                                 color=c, fontsize=fs, annotation_clip=False))
    return texts


def layout_check(fig, verbose=True):
    """Overlaps between visible text boxes, and boxes outside the figure, measured with the PGF (LaTeX) renderer."""
    import io
    from matplotlib.backends.backend_pgf import FigureCanvasPgf, RendererPgf
    FigureCanvasPgf(fig)                       # binds the PGF canvas to the figure
    r = RendererPgf(fig, io.StringIO()); fig.draw(r)   # full PGF draw so labels/ticks take their final positions
    fb = fig.bbox
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    ticklabels = set()
    for ax in fig.axes:
        ticklabels |= set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
    issues = []
    for i, (a, ba) in enumerate(texts):
        if ba.x0 < fb.x0 - 0.5 or ba.x1 > fb.x1 + 0.5 or ba.y0 < fb.y0 - 0.5 or ba.y1 > fb.y1 + 0.5:
            issues.append(("outside", a.get_text()[:40]))
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb) and not (a in ticklabels and b in ticklabels):
                issues.append(("overlap", a.get_text()[:40], b.get_text()[:40]))
    if verbose:
        print(f"layout_check: {len(texts)} texts, {len(issues)} issues")
        for it in issues[:20]:
            print("   ", it)
    return issues


def save(fig, name, inputs=(), script=None, note="", png_dpi=200):
    os.makedirs(FIGDIR, exist_ok=True); os.makedirs(os.path.join(FIGDIR, "preview"), exist_ok=True)
    pdf = os.path.join(FIGDIR, name + ".pdf")
    t0 = time.perf_counter(); fig.savefig(pdf, backend="pgf"); t_pdf = time.perf_counter() - t0
    import fitz
    doc = fitz.open(pdf); pix = doc[0].get_pixmap(dpi=png_dpi, alpha=False)
    png = os.path.join(FIGDIR, name + ".png"); pix.save(png); doc.close()
    meta = {"figure": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "script": script, "note": note,
            "size_in": list(fig.get_size_inches()), "pgf_seconds": t_pdf,
            "inputs": [{"path": os.path.relpath(p, WORK), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}
                       for p in inputs if os.path.exists(p)]}
    json.dump(meta, open(os.path.join(FIGDIR, name + ".meta.json"), "w"), indent=1)
    return pdf, png
