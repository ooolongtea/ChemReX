"""Draw the Swern-oxidation example reaction as SVG + PNG.

Image contains ONLY: reactant structure (labeled 1a) -> product structure (labeled 2a),
with reaction conditions split above/below the arrow. Compact layout, transparent
background, black & white only. The 1a/2a labels let MolDetect build a dict
(ID -> SMILES) for coreference resolution.

Outputs:
    images/reaction.svg
    images/reaction.png
"""

import io
import json
import os
import re

from rdkit.Chem import AllChem, Draw
from PIL import Image, ImageDraw, ImageFont
from rdkit.Geometry import Point2D  # noqa: E402

rdMolDraw2D = Draw.rdMolDraw2D

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = json.load(open(os.path.join(HERE, "entry.json"), encoding="utf-8"))

REACTANT_SMILES = ENTRY["smiles_ground_truth"]["1a (5-phenyl-1-pentanol)"]
PRODUCT_SMILES = ENTRY["smiles_ground_truth"]["2a (5-phenylpentanal)"]
REACTANT_ID, PRODUCT_ID = "1a", "2a"

REAGENTS = "oxalyl chloride, DMSO, Et3N"
SOLVENT = "CH2Cl2"
TEMPERATURE = "-78 C -> 25 C"
TIME = "1 h; 2 h; overnight"
YIELD = "95%"

# ── layout (px) — molecules large enough for MolDetect/RxnScribe ──
W, H = 840, 320
MOL_W, MOL_H = 250, 190
REACTANT_POS = (25, 45)
PRODUCT_POS = (565, 45)
ARROW_Y = REACTANT_POS[1] + MOL_H // 2            # 140, vertical center of molecules
ARROW_X1 = REACTANT_POS[0] + MOL_W + 15           # 290
ARROW_X2 = PRODUCT_POS[0] - 15                    # 550
ARROW_CX = (ARROW_X1 + ARROW_X2) // 2             # 420
COND_LH = 18                                     # condition line spacing
COND_ABOVE = [(REAGENTS, True), (SOLVENT, False)]       # top -> bottom (above arrow)
COND_BELOW = [(TEMPERATURE, False), (TIME, False)]      # top -> bottom (below arrow)
COND_ABOVE_BASE = ARROW_Y - 6                     # baseline of the line closest above the arrow
COND_BELOW_BASE = ARROW_Y + 20                    # baseline of the line closest below the arrow
YIELD_BASE_Y = COND_BELOW_BASE + COND_LH * len(COND_BELOW) + 6   # 202
LABEL_Y = REACTANT_POS[1] + MOL_H + 22            # 257, below each structure (close for MolDetect pairing)

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _mol(smiles):
    m = AllChem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    return m


def _opts(drawer):
    o = drawer.drawOptions()
    o.bondLineWidth = 2
    o.padding = 0.06          # small inner padding
    o.addAtomIndices = False
    o.useBWAtomPalette()      # black & white atom palette (method, not a property)
    o.clearBackground = False # transparent background (no white rect)
    return o


def render_mol_svg(smiles, w=MOL_W, h=MOL_H):
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    _opts(d)
    d.DrawMolecule(_mol(smiles))
    d.FinishDrawing()
    return d.GetDrawingText()


def render_mol_png(smiles, w=MOL_W, h=MOL_H):
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    _opts(d)
    d.DrawMolecule(_mol(smiles))
    d.FinishDrawing()
    return d.GetDrawingText()


def _inject_xy(svg_text, x, y):
    svg_text = re.sub(r"^\s*<\?xml\b.*?\?>\s*", "", svg_text, count=1, flags=re.DOTALL)
    return re.sub(r"<svg\b", f'<svg x="{x}" y="{y}"', svg_text, count=1)


def _cond_lines():
    """Return (text, bold, baseline_y) for each condition line, split above/below the arrow."""
    rows = []
    n = len(COND_ABOVE)
    for i, (text, bold) in enumerate(COND_ABOVE):       # i=0 is the top line
        y = COND_ABOVE_BASE - (n - 1 - i) * COND_LH
        rows.append((text, bold, y))
    for i, (text, bold) in enumerate(COND_BELOW):       # i=0 is closest to the arrow
        y = COND_BELOW_BASE + i * COND_LH
        rows.append((text, bold, y))
    return rows


def build_svg():
    rxn_svg = _inject_xy(render_mol_svg(REACTANT_SMILES), *REACTANT_POS)
    prod_svg = _inject_xy(render_mol_svg(PRODUCT_SMILES), *PRODUCT_POS)
    ah = 9
    arrow = (
        f'<line x1="{ARROW_X1}" y1="{ARROW_Y}" x2="{ARROW_X2-2}" y2="{ARROW_Y}" '
        f'stroke="#000" stroke-width="2.4"/>\n'
        f'<polygon points="{ARROW_X2},{ARROW_Y} {ARROW_X2-14},{ARROW_Y-ah} '
        f'{ARROW_X2-14},{ARROW_Y+ah}" fill="#000"/>\n'
    )
    cond = ""
    for text, bold, yy in _cond_lines():
        w = "bold" if bold else "normal"
        cond += (f'<text x="{ARROW_CX}" y="{yy}" text-anchor="middle" '
                 f'font-family="DejaVu Sans, Arial, sans-serif" font-size="17" '
                 f'font-weight="{w}" fill="#000">{text}</text>\n')
    yield_t = (f'<text x="{ARROW_CX}" y="{YIELD_BASE_Y}" text-anchor="middle" '
               f'font-family="DejaVu Sans, Arial, sans-serif" font-size="21" '
               f'font-weight="bold" fill="#000">{YIELD}</text>\n')
    labels = (
        f'<text x="{REACTANT_POS[0]+MOL_W//2}" y="{LABEL_Y}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Arial, sans-serif" font-size="22" '
        f'font-weight="bold" fill="#000">{REACTANT_ID}</text>\n'
        f'<text x="{PRODUCT_POS[0]+MOL_W//2}" y="{LABEL_Y}" text-anchor="middle" '
        f'font-family="DejaVu Sans, Arial, sans-serif" font-size="22" '
        f'font-weight="bold" fill="#000">{PRODUCT_ID}</text>\n'
    )
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" version="1.1">',
             rxn_svg, prod_svg, arrow, cond, yield_t, labels, "</svg>"]
    return "\n".join(parts)


def _font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
    except Exception:
        return ImageFont.load_default()


def _tw(draw, text, font):
    try:
        b = draw.textbbox((0, 0), text, font=font)
        return b[2] - b[0]
    except Exception:
        return len(text) * 12


def _center(draw, text, cx, y, font, fill):
    draw.text((cx - _tw(draw, text, font) // 2, y), text, font=font, fill=fill)


def build_png():
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))  # white bg (image models trained on white)
    draw = ImageDraw.Draw(canvas)
    mol_r = Image.open(io.BytesIO(render_mol_png(REACTANT_SMILES))).convert("RGBA")
    canvas.paste(mol_r, REACTANT_POS, mol_r)            # 3rd arg = alpha mask
    mol_p = Image.open(io.BytesIO(render_mol_png(PRODUCT_SMILES))).convert("RGBA")
    canvas.paste(mol_p, PRODUCT_POS, mol_p)
    draw.line([(ARROW_X1, ARROW_Y), (ARROW_X2, ARROW_Y)], fill="#000", width=3)
    ah = 9
    draw.polygon([(ARROW_X2, ARROW_Y), (ARROW_X2 - 14, ARROW_Y - ah),
                  (ARROW_X2 - 14, ARROW_Y + ah)], fill="#000")
    for text, bold, yy in _cond_lines():
        _center(draw, text, ARROW_CX, yy - 13, _font(17, bold=bold), "#000")
    _center(draw, YIELD, ARROW_CX, YIELD_BASE_Y - 16, _font(21, bold=True), "#000")
    _center(draw, REACTANT_ID, REACTANT_POS[0] + MOL_W // 2, LABEL_Y - 16,
            _font(22, bold=True), "#000")
    _center(draw, PRODUCT_ID, PRODUCT_POS[0] + MOL_W // 2, LABEL_Y - 16,
            _font(22, bold=True), "#000")
    return canvas


def main():
    out_dir = os.path.join(HERE, "images")
    os.makedirs(out_dir, exist_ok=True)
    svg = build_svg()
    with open(os.path.join(out_dir, "reaction.svg"), "w", encoding="utf-8") as f:
        f.write(svg)
    png = build_png()
    png.save(os.path.join(out_dir, "reaction.png"))
    print(f"SVG -> {out_dir}/reaction.svg ({len(svg)} bytes)")
    print(f"PNG -> {out_dir}/reaction.png ({os.path.getsize(out_dir + '/reaction.png')} bytes)")


if __name__ == "__main__":
    main()
