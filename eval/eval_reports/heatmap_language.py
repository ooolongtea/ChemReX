#!/usr/bin/env python3
"""
Cross-lingual NER F1 heatmap (EN | ZH | ZH-EN gap) for IEEE double-column.
Replaces the radar chart in radar_language.py.

Layout: rows = models, cols = entity types.
Panel 1 (EN) and Panel 2 (ZH) share one sequential scale so absolute
performance is directly comparable. Panel 3 (ZH - EN) is a diverging map
centered at 0 -> cells near white = language-balanced (the key message).
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

# ============================================================
# Data sources
# ============================================================
_results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
JSON_FILES = [
    ("Qwen3-8B", os.path.join(_results, "external_Qwen3-8B_20260630-115257", "eval_metrics_language.json")),
    ("DeepSeek-V3", os.path.join(_results, "external_deepseek-v3_20260201-155435", "eval_metrics_language.json")),
    ("GPT-4o", os.path.join(_results, "external_gpt-4o_20260131-205535", "eval_metrics_language.json")),
    ("Qwen-Plus", os.path.join(_results, "external_qwen-plus_20260201-145935", "eval_metrics_language.json")),
    ("ChemReX", os.path.join(_results, "external_qwen3-lora-08_20260630-121627", "eval_metrics_language.json")),
    ("ReactionSeek", os.path.join(_results, "reactionseek_Qwen3-8B_20260701_174918", "eval_metrics_language.json")),
]
OURS = {"ChemReX", "ReactionSeek"}
OUTPUT_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap_language")

FIELDS = ['REACTION_PRODUCT', 'STARTING_MATERIAL', 'REAGENT_CATALYST',
          'SOLVENT', 'TEMPERATURE', 'TIME', 'YIELD_PERCENT']
FIELD_LABELS = ['Prod.', 'Start.\nMat.', 'Rea./\nCat.', 'Solv.', 'Temp.', 'Time', 'Yield']

# ============================================================
# Publication style (IEEE double-column, editable vector text)
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.6

SEQ_CMAP = 'viridis'
DIV_CMAP = 'RdBu_r'
ANNOT_FS = 5.6
LABEL_FS = 7
TITLE_FS = 8


def load_matrix():
    """Return (models, en[M,F], zh[M,F]) as F1 in percent."""
    models, en_rows, zh_rows = [], [], []
    for name, path in JSON_FILES:
        if not os.path.exists(path):
            print(f"! missing: {path}")
            continue
        data = json.load(open(path, encoding='utf-8'))
        bl = data.get('by_language', {})

        def row(lang):
            fm = bl.get(lang, {}).get('ner', {}).get('field_metrics', {})
            return [100.0 * fm.get(f, {}).get('f1', 0.0) for f in FIELDS]

        models.append(name)
        en_rows.append(row('EN'))
        zh_rows.append(row('ZH'))
    return models, np.array(en_rows), np.array(zh_rows)


def _annotate(ax, mat, fmt, seq_norm=None, cmap=None, signed=False):
    for (i, j), v in np.ndenumerate(mat):
        if seq_norm is not None and cmap is not None:
            r, g, b, _ = plt.get_cmap(cmap)(seq_norm(v))
            txt_color = 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else 'black'
        else:
            txt_color = 'black'
        s = f"{v:+.0f}" if signed else f"{v:.0f}"
        ax.text(j, i, s, ha='center', va='center', fontsize=ANNOT_FS, color=txt_color)


def _style_axis(ax, models, xlabels, show_y):
    ax.set_xticks(range(len(FIELDS)))
    ax.set_xticklabels(xlabels, fontsize=LABEL_FS)
    ax.set_yticks(range(len(models)))
    if show_y:
        ax.set_yticklabels(models, fontsize=LABEL_FS)
        for lab, m in zip(ax.get_yticklabels(), models):
            if m in OURS:
                lab.set_fontweight('bold')
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # thin white gridlines between cells
    ax.set_xticks(np.arange(-.5, len(FIELDS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', length=0)


def main():
    models, en, zh = load_matrix()
    gap = zh - en

    vmin = np.floor(min(en.min(), zh.min()) / 5) * 5
    vmax = 100.0
    seq_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    glim = max(5.0, np.abs(gap).max())
    div_norm = TwoSlopeNorm(vmin=-glim, vcenter=0.0, vmax=glim)

    fig = plt.figure(figsize=(7.16, 2.55))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1, 0.05],
        width_ratios=[1, 1, 1],
        left=0.11, right=0.995, top=0.86, bottom=0.20,
        wspace=0.10, hspace=0.55,
    )
    axEN = fig.add_subplot(gs[0, 0])
    axZH = fig.add_subplot(gs[0, 1])
    axGP = fig.add_subplot(gs[0, 2])

    imEN = axEN.imshow(en, cmap=SEQ_CMAP, norm=seq_norm, aspect='auto')
    axZH.imshow(zh, cmap=SEQ_CMAP, norm=seq_norm, aspect='auto')
    imGP = axGP.imshow(gap, cmap=DIV_CMAP, norm=div_norm, aspect='auto')

    _annotate(axEN, en, "{:.0f}", seq_norm, SEQ_CMAP)
    _annotate(axZH, zh, "{:.0f}", seq_norm, SEQ_CMAP)
    _annotate(axGP, gap, "{:+.0f}", signed=True)

    _style_axis(axEN, models, FIELD_LABELS, show_y=True)
    _style_axis(axZH, models, FIELD_LABELS, show_y=False)
    _style_axis(axGP, models, FIELD_LABELS, show_y=False)

    axEN.set_title("a  English F1", fontsize=TITLE_FS, fontweight='bold', loc='left', pad=4)
    axZH.set_title("b  Chinese F1", fontsize=TITLE_FS, fontweight='bold', loc='left', pad=4)
    axGP.set_title("c  ZH \u2212 EN gap", fontsize=TITLE_FS, fontweight='bold', loc='left', pad=4)

    # shared colorbar for panels a,b
    cax1 = fig.add_subplot(gs[1, 0:2])
    cb1 = fig.colorbar(imEN, cax=cax1, orientation='horizontal')
    cb1.set_label("F1 (%)", fontsize=LABEL_FS, labelpad=1)
    cb1.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb1.outline.set_linewidth(0.5)

    # colorbar for gap panel
    cax2 = fig.add_subplot(gs[1, 2])
    cb2 = fig.colorbar(imGP, cax=cax2, orientation='horizontal')
    cb2.set_label("F1 points (ZH \u2212 EN)", fontsize=LABEL_FS, labelpad=1)
    cb2.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb2.outline.set_linewidth(0.5)

    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{OUTPUT_PREFIX}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"saved: {OUTPUT_PREFIX}.[svg|pdf|png]")

    # quick language-balance summary (mean |gap| per model)
    print("\nmean |ZH-EN| gap (F1 pts):")
    for m, g in zip(models, np.abs(gap).mean(axis=1)):
        print(f"  {m:14s} {g:5.1f}")


if __name__ == "__main__":
    main()
