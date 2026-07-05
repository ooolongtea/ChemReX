#!/usr/bin/env python3
"""
Cross-lingual field-level F1 heatmap (EN | ZH | ZH-EN gap) for IEEE BIBM.
Replaces the radar chart.

Reads field-level NER F1 directly from the per-language evaluation JSONs on
this server (no numbers are fabricated). Three horizontal panels:
  (a) English F1     - sequential colormap, fixed range 30-100
  (b) Chinese F1     - same colormap, same range 30-100
  (c) ZH-EN dF1      - diverging colormap centered at 0, range [-25, 25],
                       plus a final "Mean |d|" column that summarizes each
                       model's average absolute cross-lingual gap.
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

# ------------------------------------------------------------------
# Data sources (existing experimental results on the server)
# ------------------------------------------------------------------
_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
JSON_FILES = [
    ("ChemLLM-7B-SFT",   os.path.join(_results, "base_ChemLLM-7B-SFT_20260130-155158",   "eval_metrics_language.json")),
    ("ChemDFM-v1.5-8B",  os.path.join(_results, "base_chemdfm-v1.5-8b_20260128-154402",   "eval_metrics_language.json")),
    ("GPT-4o",           os.path.join(_results, "external_gpt-4o_20260131-205535",        "eval_metrics_language.json")),
    ("DeepSeek-V4-Pro",  os.path.join(_results, "external_deepseek-v4-pro_20260704-225238","eval_metrics_language.json")),
    ("Qwen3-8B\n+ ReactionSeek",os.path.join(_results, "reactionseek_Qwen3-8B_20260701_174918", "eval_metrics_language.json")),
    ("ChemReX",          os.path.join(_results, "external_qwen3-lora-08_20260630-121627", "eval_metrics_language.json")),
]
BOLD_ROWS = {"ChemReX"}
OUTPUT_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_lingual_heatmap")

FIELDS = ['REACTION_PRODUCT', 'STARTING_MATERIAL', 'REAGENT_CATALYST',
          'SOLVENT', 'TEMPERATURE', 'TIME', 'YIELD_PERCENT']
FIELD_LABELS = ['Prod.', 'SM', 'Reag.', 'Solv.', 'Temp.', 'Time', 'Yield']

# fixed color scales requested for reproducibility
SEQ_VMIN, SEQ_VMAX = 30.0, 100.0
DIV_LIM = 25.0
SEQ_CMAP = 'viridis'
DIV_CMAP = 'RdBu_r'

# ------------------------------------------------------------------
# Publication style (IEEE double-column, editable vector text)
# ------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.linewidth': 0.6,
})
ANNOT_FS = 5.8
LABEL_FS = 7
TITLE_FS = 8


def load_matrix():
    """Return (models, en[M,F], zh[M,F]) as field-level F1 in percent."""
    models, en_rows, zh_rows = [], [], []
    for name, path in JSON_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = json.load(open(path, encoding='utf-8'))
        bl = data.get('by_language', {})

        def row(lang):
            fm = bl.get(lang, {}).get('ner', {}).get('field_metrics', {})
            return [100.0 * fm.get(f, {}).get('f1', 0.0) for f in FIELDS]

        models.append(name)
        en_rows.append(row('EN'))
        zh_rows.append(row('ZH'))
    return models, np.array(en_rows), np.array(zh_rows)


def _txt_color(cmap, norm, v):
    r, g, b, _ = plt.get_cmap(cmap)(norm(v))
    return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else 'black'


def _seq_panel(ax, mat, models, norm, show_y, title):
    ax.imshow(mat, cmap=SEQ_CMAP, norm=norm, aspect='auto')
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.0f}", ha='center', va='center',
                fontsize=ANNOT_FS, color=_txt_color(SEQ_CMAP, norm, v))
    _style_axis(ax, models, FIELD_LABELS, show_y)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left', pad=4)
    return ax.images[0]


def _gap_panel(ax, gap, mean_abs, models, norm, title):
    """Gap columns + a final Mean |d| column, all on the diverging norm."""
    mat = np.hstack([gap, mean_abs[:, None]])
    ax.imshow(mat, cmap=DIV_CMAP, norm=norm, aspect='auto')
    n_gap = gap.shape[1]
    for (i, j), v in np.ndenumerate(mat):
        if j < n_gap:
            s = f"{v:+.0f}"
        else:
            s = f"{v:.0f}"          # Mean |d| column: absolute magnitude
        ax.text(j, i, s, ha='center', va='center',
                fontsize=ANNOT_FS, color=_txt_color(DIV_CMAP, norm, v))
    labels = FIELD_LABELS + ['Mean\n|\u0394|']
    _style_axis(ax, models, labels, show_y=False, ncol=mat.shape[1])
    # visual separator before the summary column
    ax.axvline(n_gap - 0.5, color='0.25', linewidth=1.0)
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', loc='left', pad=4)
    return ax.images[0]


def _style_axis(ax, models, xlabels, show_y, ncol=None):
    ncol = len(xlabels) if ncol is None else ncol
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=LABEL_FS)
    ax.set_yticks(range(len(models)))
    if show_y:
        ax.set_yticklabels(models, fontsize=LABEL_FS)
        for lab, m in zip(ax.get_yticklabels(), models):
            if m in BOLD_ROWS:
                lab.set_fontweight('bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', length=0)


def _print_table(title, models, mat, signed=False):
    print(f"\n{title}")
    header = "  {:18s}".format("Model") + "".join(f"{h:>7s}" for h in FIELD_LABELS)
    print(header)
    for m, row in zip(models, mat):
        cells = "".join((f"{v:+7.0f}" if signed else f"{v:7.1f}") for v in row)
        print(f"  {m.replace(chr(10), ' '):22s}{cells}")


def main():
    models, en, zh = load_matrix()
    gap = zh - en
    mean_abs = np.abs(gap).mean(axis=1)

    seq_norm = Normalize(vmin=SEQ_VMIN, vmax=SEQ_VMAX)
    div_norm = TwoSlopeNorm(vmin=-DIV_LIM, vcenter=0.0, vmax=DIV_LIM)

    fig = plt.figure(figsize=(7.16, 2.55), facecolor='white')
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1, 0.05],
        width_ratios=[1, 1, 1.16],
        left=0.115, right=0.995, top=0.86, bottom=0.20,
        wspace=0.12, hspace=0.55,
    )
    axEN = fig.add_subplot(gs[0, 0])
    axZH = fig.add_subplot(gs[0, 1])
    axGP = fig.add_subplot(gs[0, 2])

    imEN = _seq_panel(axEN, en, models, seq_norm, True,  "(a) English F1")
    _seq_panel(axZH, zh, models, seq_norm, False, "(b) Chinese F1")
    imGP = _gap_panel(axGP, gap, mean_abs, models, div_norm, "(c) ZH\u2013EN \u0394F1")

    # shared colorbar for panels a,b
    cax1 = fig.add_subplot(gs[1, 0:2])
    cb1 = fig.colorbar(imEN, cax=cax1, orientation='horizontal')
    cb1.set_label("F1 (%)", fontsize=LABEL_FS, labelpad=1)
    cb1.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb1.outline.set_linewidth(0.5)

    # colorbar for gap panel
    cax2 = fig.add_subplot(gs[1, 2])
    cb2 = fig.colorbar(imGP, cax=cax2, orientation='horizontal')
    cb2.set_label("\u0394F1 points", fontsize=LABEL_FS, labelpad=1)
    cb2.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb2.outline.set_linewidth(0.5)

    fig.savefig(f"{OUTPUT_PREFIX}.pdf", bbox_inches='tight')
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {OUTPUT_PREFIX}.[pdf|png]")

    # ---- console verification of exact input values ----
    _print_table("English F1 (%)", models, en)
    _print_table("Chinese F1 (%)", models, zh)
    _print_table("ZH-EN dF1 (points)", models, gap, signed=True)
    print("\nMean |dF1| by model (F1 points):")
    for m, g in zip(models, mean_abs):
        print(f"  {m.replace(chr(10), ' '):22s} {g:5.1f}")


if __name__ == "__main__":
    main()
