import json
import unittest
from pathlib import Path

from pipeline.document.segmenter import fallback_procedures, load_blocks, segment_blocks


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/md/US9732058B2/vlm/US9732058B2_content_list.json"


class DocumentSegmenterTest(unittest.TestCase):
    def test_repeated_heading_suffix_cannot_replace_a_distinct_numbered_heading(self):
        for headings in (["Example C"] * 5 + ["Example C-5"],
                         ["Example C-5"] + ["Example C"] * 5,
                         ["Step B", "Step B", "Step B-2"]):
            with self.subTest(headings=headings):
                blocks = []
                for page, heading in enumerate(headings):
                    blocks.extend([
                        {"type": "text", "text": heading, "page_idx": page},
                        {"type": "text", "text": "Compound A (2 g) was stirred in ethanol for 2 h.",
                         "page_idx": page},
                    ])
                chunks = segment_blocks(blocks, "sample.json")
                ids = [chunk["id"] for chunk in chunks]
                self.assertEqual(len(chunks), len(headings))
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual([chunk["page_start"] for chunk in chunks],
                                 list(range(1, len(headings) + 1)))

    def test_english_and_chinese_heading_variants(self):
        for heading in ("Example 1", "Example 1. Compound A",
                        "Example 1: Compound A", "实施例 1：化合物甲",
                        "实施例 1 化合物甲"):
            blocks = [{"type": "text", "text": heading, "page_idx": 0},
                      {"type": "text", "text": "The mixture was stirred for 2 h.",
                       "page_idx": 0}]
            chunks = segment_blocks(blocks, "sample.json")
            self.assertEqual(len(chunks), 1, heading)
            self.assertEqual(chunks[0]["kind"], "example", heading)
            self.assertTrue(chunks[0]["has_procedure"], heading)

    def test_fallback_requires_operation_and_context(self):
        blocks = [{"type": "text", "text": "Claims: a composition containing 3 mg drug.",
                   "page_idx": 0},
                  {"type": "text", "text":
                   "A mixture of compound A (2 g) and ethanol (5 mL) was stirred for 2 h.",
                   "page_idx": 1}]
        chunks = fallback_procedures(blocks, "sample.json")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_start"], 2)

    def test_numbered_claim_boundary(self):
        blocks = [
            {"type": "text", "text": "[0101] 实施例一：甲", "page_idx": 0},
            {"type": "text", "text": "甲(2g)加入乙醇并加热2小时，得到产物。", "page_idx": 0},
            {"type": "text", "text": "[0102] We claim:", "page_idx": 0},
            {"type": "text", "text": "A preparation comprising 3 mg compound.",
             "page_idx": 0},
        ]
        chunks = segment_blocks(blocks, "sample.json")
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("claim", chunks[0]["text"].lower())

    def test_header_example_and_biological_boundary(self):
        blocks = [
            {"type": "header", "text": "Example 120", "page_idx": 0},
            {"type": "text", "text": "Using an analogous method, compound B was obtained from compound A.", "page_idx": 0},
            {"type": "header", "text": "Example 122", "page_idx": 1},
            {"type": "text", "text": "Compound C was obtained from compound B.", "page_idx": 1},
            {"type": "text", "text": "Biological Activity", "page_idx": 1},
            {"type": "text", "text": "Cells were tested in a 5 mg assay.", "page_idx": 2},
        ]
        chunks = segment_blocks(blocks, "sample.json")
        self.assertEqual([c["id"] for c in chunks], ["example_120", "example_122", "biological_activity_0"])
        self.assertEqual(chunks[-1]["process_kind"], "assay")
        self.assertFalse(chunks[0]["candidate_evidence"])
        self.assertNotIn("Cells were tested", chunks[1]["text"])

    def test_late_two_column_headers_are_reordered_with_original_indices(self):
        blocks = [
            {"type": "text", "text": "Left product", "page_idx": 0,
             "bbox": [100, 150, 350, 180]},
            {"type": "text", "text": "Right product", "page_idx": 0,
             "bbox": [550, 150, 800, 180]},
            {"type": "text", "text": "Left product (2 g) was stirred for 2 h.",
             "page_idx": 0, "bbox": [100, 300, 350, 380]},
            {"type": "text", "text": "Right product (3 g) was heated for 1 h.",
             "page_idx": 0, "bbox": [550, 300, 800, 380]},
            {"type": "header", "text": "Example 120", "page_idx": 0,
             "bbox": [150, 100, 250, 120]},
            {"type": "header", "text": "Example 122", "page_idx": 0,
             "bbox": [600, 100, 700, 120]},
        ]
        chunks = segment_blocks(blocks, "sample.json")
        self.assertEqual([c["id"] for c in chunks], ["example_120", "example_122"])
        self.assertEqual(chunks[0]["block_indices"], [4, 0, 2])
        self.assertEqual(chunks[1]["block_indices"], [5, 1, 3])

    def test_unnumbered_preparation_and_late_fallback(self):
        blocks = [{"type": "text", "text": "General background.", "page_idx": 0}]
        blocks += [{"type": "text", "text":
                    f"Compound {i} (2 g) was stirred in ethanol (5 mL) for 2 h.",
                    "page_idx": i} for i in range(20)]
        chunks = fallback_procedures(blocks, "sample.json")
        self.assertTrue(any(c["block_end"] >= 20 for c in chunks))
        self.assertTrue(all(len(c["text"]) <= 6000 for c in chunks))
        titled = segment_blocks([
            {"type": "text", "text": "Preparation of Compound X", "page_idx": 0},
            {"type": "text", "text": "Compound X (2 g) was stirred for 2 h.", "page_idx": 0},
        ], "sample.json")
        self.assertEqual(titled[0]["kind"], "preparation")

    def test_fallback_splits_long_block_without_dropping_tail(self):
        text = "Compound A (2 g) was stirred in ethanol (5 mL). " * 300
        chunks = fallback_procedures([
            {"type": "text", "text": text, "page_idx": 0}], "sample.json")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"]) <= 6000 for c in chunks))
        self.assertEqual(" ".join(c["text"] for c in chunks), text.strip())

    def test_patent_structure_and_cross_page_procedure(self):
        if not SOURCE.exists():
            self.skipTest("patent fixture unavailable")
        chunks = segment_blocks(load_blocks(str(SOURCE)), SOURCE.name)
        by_id = {chunk["id"]: chunk for chunk in chunks}
        self.assertEqual(len(chunks), 244)
        self.assertEqual(sum(c["kind"] == "example" for c in chunks), 65)
        self.assertEqual(sum(c["kind"] == "intermediate" for c in chunks), 179)
        self.assertEqual((by_id["example_3"]["page_start"], by_id["example_3"]["page_end"]), (40, 41))
        self.assertIn("23 mg, 15%", by_id["example_3"]["procedure"])
        self.assertNotIn("NMR", by_id["example_3"]["procedure"])
        self.assertIn("70 mg, 38%", by_id["example_1"]["procedure"])
        self.assertIn("intermediate_39", by_id["intermediate_38"]["references"])
        self.assertTrue(all(f"example_28a_p{n}" in by_id for n in (1, 2, 3)))
        self.assertIn("intermediate_43a", by_id)
        self.assertIn("intermediate_130_2", by_id)

    def test_reference_example_heading_switches_parent_scope(self):
        blocks = [
            {"type": "text", "text": "Example 1", "page_idx": 0},
            {"type": "text", "text": "Step 1", "page_idx": 0},
            {"type": "text", "text": "Compound A (2 g) was stirred in ethanol for 2 h.", "page_idx": 0},
            {"type": "text", "text": "Reference Example 2", "page_idx": 1},
            {"type": "text", "text": "Step 1", "page_idx": 1},
            {"type": "text", "text": "Compound B (2 g) was stirred in ethanol for 2 h.", "page_idx": 1},
        ]
        chunks = segment_blocks(blocks, "sample.json")
        steps = [c for c in chunks if c["kind"] == "step"]
        self.assertEqual([c["parent_example_id"] for c in steps],
                         ["example_1", "reference_example_2"])

    def test_chinese_reference_does_not_merge_arabic_and_cjk_numerals(self):
        blocks = [
            {"type": "text", "text": "实施例1", "page_idx": 0},
            {"type": "text", "text": "甲(2g)加入乙醇并加热2小时，得到产物。与实施例1一致。",
             "page_idx": 0},
        ]
        chunks = segment_blocks(blocks, "sample.json")
        self.assertEqual(len(chunks), 1)
        self.assertIn("example_1", chunks[0]["references"])

    def test_decimal_subexample_reference_preserves_letter_suffix(self):
        blocks = [
            {"type": "text", "text": "Example 1", "page_idx": 0},
            {"type": "text", "text": "Intermediate 223.1a was converted to Intermediate "
                                     "223.2 according to the procedure of Example 170.",
             "page_idx": 0},
        ]
        chunks = segment_blocks(blocks, "sample.json")
        self.assertEqual(len(chunks), 1)
        self.assertIn("intermediate_223.1a", chunks[0]["references"])
        self.assertIn("intermediate_223.2", chunks[0]["references"])

    def test_scheme_caption_product_headings_split_five_transformations(self):
        """US8598181 ex.18: four printed products are image annotations.

        Only captions whose following procedure reports obtaining that compound
        may head a step; the reactant label in the scheme must stay body text.
        """
        from pipeline.document.step_context import split_steps
        names = [
            "3-Chloro-4,5-dihydroxybenzaldehyde",
            "7-Chlorobenzo[d][1,3]dioxole-5-carbaldehyde",
            "(7-Chlorobenzo[d][1,3]dioxol-5-yl)methanol",
            "4-Chloro-6-(chloromethyl)benzo[d][1,3]dioxole",
            "2-(7-Chlorobenzo[d][1,3]dioxol-5-yl)acetonitrile",
        ]
        procedures = [
            "To a suspension of 3-chloro-4-hydroxy-5-methoxy-benzaldehyde (10 g, 54 mmol) "
            "in dichloromethane was added BBr3 (26.7 g, 107 mmol) at -40 C.",
            "After the addition the mixture was stirred for 5 h and the filtrate was "
            "evaporated under reduced pressure to afford 3-chloro-4,5-dihydroxybenzaldehyde "
            "(9.8 g, 89%), which was directly used in the next step.",
            "To a solution of 3-chloro-4,5-dihydroxybenzaldehyde (8.0 g, 46 mmol) and "
            "BrClCH2 in dry DMF was added Cs2CO3. The mixture was stirred at 60 C overnight "
            "to afford 7-chlorobenzo[d][1,3]dioxole-5-carbaldehyde (6.0 g, 70%).",
            "To a solution of 7-chlorobenzo[d][1,3]dioxole-5-carbaldehyde (6.0 g, 33 mmol) "
            "in THF was added NaBH4 (2.5 g, 64 mmol). The mixture was stirred for 30 min to "
            "afford (7-chlorobenzo[d][1,3]dioxol-5-yl)methanol.",
            "A mixture of (7-chlorobenzo[d][1,3]-dioxol-5-yl)methanol (5.5 g, 30 mmol) and "
            "SOCl2 in dichloromethane was stirred for 1 h to afford "
            "4-chloro-6-(chloromethyl)benzo[d][1,3]dioxole.",
            "A mixture of 4-chloro-6-(chloromethyl)benzo[d][1,3]dioxole (6.0 g, 29 mmol) and "
            "NaCN in DMSO was stirred at 40 C for 1 h to afford "
            "2-(7-chlorobenzo[d][1,3]dioxol-5-yl)acetonitrile (3.4 g, 58%).",
        ]
        parts = [
            dict(text=names[0], block_idx=1, page_idx=0, source_role="image_annotation"),
            dict(text=procedures[0], block_idx=2, page_idx=0),
            dict(text=procedures[1], block_idx=3, page_idx=0),
            dict(text=names[1], block_idx=4, page_idx=0, source_role="image_annotation"),
            dict(text=procedures[2], block_idx=5, page_idx=0),
            dict(text=names[2], block_idx=6, page_idx=0, source_role="image_annotation"),
            dict(text=procedures[3], block_idx=7, page_idx=0),
            dict(text=names[3], block_idx=8, page_idx=0, source_role="image_annotation"),
            dict(text=procedures[4], block_idx=9, page_idx=0),
            dict(text=names[4], block_idx=10, page_idx=0),
            dict(text=procedures[5], block_idx=11, page_idx=0),
        ]
        chunk = dict(id="example_18", kind="example", title=names[4],
                     text=names[4] + "\n" + "\n".join(p["text"] for p in parts),
                     text_parts=[p["text"] for p in parts], source_parts=parts)
        steps = split_steps(chunk)
        self.assertEqual([step["title"] for step in steps], names)
        self.assertEqual(len(steps), 5)

    def test_numeric_chunk_title_is_backfilled_from_compound_heading(self):
        from pipeline.document.step_context import attach_context
        compound = "6-氨基甲酰基-4-(3-甲基-4-硝基苯氧基)-7-甲氧基喹啉"
        procedure = ("与制备例 7 同样，由 7-甲氧基-4-氯-喹啉-6-甲酰胺(1.0g)和 "
                     "4-硝基-3-甲基苯酚(810mg)得到标题化合物(1.2g)。")
        blocks = [
            {"type": "text", "text": "实施例 488-1", "page_idx": 0},
            {"type": "text", "text": compound, "page_idx": 0},
            {"type": "text", "text": procedure, "page_idx": 0},
        ]
        chunk = dict(id="example_488_2", reference_id="example_488", kind="example", title="1",
                     source="sample.json", page_start=1, page_end=1, block_start=1, block_end=2,
                     block_indices=[1, 2], images=[], tables=[], has_procedure=True,
                     procedure=procedure, procedure_raw=procedure,
                     text="1\n" + compound + "\n" + procedure,
                     text_parts=[compound, procedure],
                     title_source={"text": "1", "block_idx": 0, "page_idx": 0})
        result = attach_context([chunk], blocks, "sample.json")[0]
        self.assertEqual(result["title"], compound)
        self.assertEqual(result["title_source"]["text"], compound)
        self.assertEqual(result["title_source"]["block_idx"], 1)
        # chunk_processing compares the terminal step title with the chunk title.
        self.assertEqual(result["steps"][-1]["title"], result["title"])

    def test_preparative_example_composite_reference_resolves(self):
        from pipeline.document.step_context import _reference_targets, resolve_programs
        plain = dict(id="example_141", kind="example", source="sample.json", chunk_id="example_141",
                     blocks=[dict(block_idx=10, page_idx=0, text="实施例 141")],
                     text="实施例 141\n加入甲醇，搅拌得到产物。")
        preparative = dict(id="example_141", kind="example", source="sample.json",
                           chunk_id="example_141_2",
                           blocks=[dict(block_idx=11, page_idx=0, text="制备例 141 - 1")],
                           text="1\n向 4-(6-氨基甲酰基-7-甲氧基-4-喹啉基)氧-2-甲基苯胺"
                                "中加入试剂，搅拌 2 小时，得到标题化合物。")
        index = {"example_141": [plain, preparative]}
        for text in ("按照制备例 141-1 记载的方法，由 4-(...)得到标题化合物。",
                     "The title compound was prepared as described in "
                     "Preparative Example 141-1."):
            self.assertEqual(_reference_targets(text)[0], ["example_141_1"])
            refs, issues = resolve_programs(text, index)
            self.assertFalse(issues)
            self.assertEqual([(ref["id"], ref["status"]) for ref in refs],
                             [("example_141", "resolved")])
            # The printed composite heading selects the cited sub-unit, not the
            # unrelated plain Example 141 that parsed to the same id.
            self.assertEqual(refs[0]["chunk_id"], "example_141_2")
        # "与制备例 X 同样" states the citation before its cue.
        same_manner = {"example_7": [dict(preparative, id="example_7", chunk_id="example_7")]}
        self.assertEqual(_reference_targets("与制备例 7 同样，由原料得到产物。")[0],
                         ["example_7"])
        refs, issues = resolve_programs("与制备例 7 同样，由原料得到产物。", same_manner)
        self.assertFalse(issues)
        self.assertEqual([ref["status"] for ref in refs], ["resolved"])


if __name__ == "__main__":
    unittest.main()
