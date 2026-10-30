"""Fallback image-window regressions (2026-10-27): a scheme figure split by
the VLM into consecutive image blocks must attach whole to the fallback
chunk that describes it."""

from pipeline.document.segmenter import fallback_procedures

PROCEDURE = ("[0097] 将化合物A(2.0克,12.3毫摩尔)溶于乙醇(50mL)，加热至80℃搅拌2小时，"
             "TLC显示反应完成，得到标题化合物(1.5克，产率80%)。")
TAIL = "[0098] 方案B为合成的一般方法，其中各基团如前定义。"


def image_block(name):
    return {"type": "image", "img_path": f"images/{name}", "page_idx": 0}


def test_contiguous_split_scheme_figure_attaches_whole():
    # Reproduces the CN_ZL201711187449.3 page-21 layout: the scheme figure
    # was split into two image blocks before and after the describing text.
    blocks = [
        image_block("scheme_a_part1.jpg"),
        image_block("scheme_a_part2.jpg"),
        {"type": "text", "text": PROCEDURE, "page_idx": 0},
        {"type": "text", "text": TAIL, "page_idx": 0},
        image_block("scheme_b_part1.jpg"),
        image_block("scheme_b_part2.jpg"),
        {"type": "text", "text": "References", "page_idx": 0},
    ]
    chunks = fallback_procedures(blocks, "sample.json")
    assert len(chunks) == 1
    assert chunks[0]["images"] == ["images/scheme_a_part1.jpg", "images/scheme_a_part2.jpg",
                                   "images/scheme_b_part1.jpg", "images/scheme_b_part2.jpg"]


def test_expansion_stops_at_non_image_block():
    # An image separated from the window by unrelated text stays unattached.
    blocks = [
        image_block("far_figure.jpg"),
        {"type": "text", "text": "Background paragraph without any procedure.",
         "page_idx": 0},
        image_block("near_figure.jpg"),
        {"type": "text", "text": PROCEDURE, "page_idx": 0},
        {"type": "text", "text": "References", "page_idx": 0},
    ]
    chunks = fallback_procedures(blocks, "sample.json")
    assert len(chunks) == 1
    assert chunks[0]["images"] == ["images/near_figure.jpg"]


def test_single_adjacent_image_still_attaches():
    blocks = [
        image_block("lone.jpg"),
        {"type": "text", "text": PROCEDURE, "page_idx": 0},
        {"type": "text", "text": "References", "page_idx": 0},
    ]
    chunks = fallback_procedures(blocks, "sample.json")
    assert len(chunks) == 1
    assert chunks[0]["images"] == ["images/lone.jpg"]
