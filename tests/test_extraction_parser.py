"""Chemical names with locant commas must remain one compound."""

import json
import unittest

from pipeline.extraction.parser import (
    _split_multi_value, parse_action_details, parse_ner_records,
)


class MultiValueSplitTests(unittest.TestCase):
    def test_preserves_locants_inside_chemical_name(self):
        product = "N-{4-methoxy-2-[1-methyl-3,6-dihydro-2H-pyridin-4-yl]}amide"
        self.assertEqual(_split_multi_value(product), [product])

    def test_splits_separate_reagents(self):
        self.assertEqual(_split_multi_value("Acryloyl chloride, DIPEA"),
                         ["Acryloyl chloride", "DIPEA"])

    def test_preserves_letter_locant_comma(self):
        self.assertEqual(_split_multi_value("N, N-dimethylformamide"),
                         ["N, N-dimethylformamide"])
        self.assertEqual(_split_multi_value("HATU, N, N-diisopropylethylamine"),
                         ["HATU", "N, N-diisopropylethylamine"])

    def test_still_splits_separate_letter_reagents(self):
        self.assertEqual(_split_multi_value("A, B"), ["A", "B"])


class NERListSplitTests(unittest.TestCase):
    def test_crystal_form_suffix_stays_on_the_compound(self):
        parent = "N-(2,6-dimethylphenyl)acetamide"
        records = parse_ner_records(json.dumps({
            "REACTION_PRODUCT": [parent + ", Form C"],
            "STARTING_MATERIAL": [parent + ", Form A"],
        }))
        self.assertEqual(records[0]["REACTION_PRODUCT"], [parent + ", Form C"])
        self.assertEqual(records[0]["STARTING_MATERIAL"], [parent + ", Form A"])
        self.assertEqual(_split_multi_value(parent + ", Form C, ethanol"),
                         [parent + ", Form C", "ethanol"])

    def test_splits_semicolon_packed_list_items(self):
        records = parse_ner_records(json.dumps({
            "REACTION_PRODUCT": ["化合物A;化合物B"],
            "STARTING_MATERIAL": ["C；D", "C"],
            "SOLVENT": {"reaction": ["甲苯；乙醇"]},
        }, ensure_ascii=False))
        self.assertEqual(records[0]["REACTION_PRODUCT"], ["化合物A", "化合物B"])
        self.assertEqual(records[0]["STARTING_MATERIAL"], ["C", "D"])
        self.assertEqual(records[0]["SOLVENT"], ["甲苯", "乙醇"])

    def test_keeps_locant_commas_inside_list_items(self):
        product = "N-{4-methoxy-2-[1-methyl-3,6-dihydro-2H-pyridin-4-yl]}amide"
        records = parse_ner_records(json.dumps({"REACTION_PRODUCT": [product]}))
        self.assertEqual(records[0]["REACTION_PRODUCT"], [product])

    def test_list_item_preserves_letter_locant_comma(self):
        records = parse_ner_records(json.dumps(
            {"REAGENT_CATALYST": ["HATU, N, N-diisopropylethylamine"]}))
        self.assertEqual(records[0]["REAGENT_CATALYST"],
                         ["HATU", "N, N-diisopropylethylamine"])

    def test_scalar_list_join_is_unchanged(self):
        records = parse_ner_records(json.dumps({"TEMPERATURE": ["室温", "50°C"]},
                                               ensure_ascii=False))
        self.assertEqual(records[0]["TEMPERATURE"], "室温; 50°C")


class ActionLoopTests(unittest.TestCase):
    def test_truncates_repeated_cycle(self):
        actions = ["加入", "搅拌", "回流", "冷却", "除去"] * 10
        details = parse_action_details(",".join(actions))
        self.assertLessEqual(len(details["actions"]), 10)
        self.assertEqual(details["actions"],
                         ["加入", "搅拌", "回流", "冷却", "除去"] * 2)
        self.assertEqual(details["status"], "ok")
        self.assertEqual(details["loop_truncated"]["period"], 5)
        self.assertEqual(details["loop_truncated"]["repeats"], 10)

    def test_keeps_two_legitimate_repeats(self):
        actions = ["加入", "搅拌", "加入", "搅拌"]
        details = parse_action_details(",".join(actions))
        self.assertEqual(details["actions"], actions)
        self.assertNotIn("loop_truncated", details)

    def test_keeps_four_legitimate_cycles(self):
        # Portionwise additions can legitimately repeat a 2-action cycle four
        # times; only clearly degenerate tails (>= 8 repeats) are truncated.
        actions = ["加入", "搅拌"] * 4
        details = parse_action_details(",".join(actions))
        self.assertEqual(details["actions"], actions)
        self.assertNotIn("loop_truncated", details)

    def test_collapses_consecutive_duplicates(self):
        details = parse_action_details("加入,加入,搅拌,搅拌")
        self.assertEqual(details["actions"], ["加入", "搅拌"])
        self.assertEqual(details["loop_truncated"]["duplicates_removed"], 2)

    def test_caps_pathological_length(self):
        # A 9-step cycle is longer than max_period, so it is not treated as a
        # loop; the absolute cap still bounds the accepted sequence.
        cycle = ["加入", "搅拌", "回流", "冷却", "除去",
                 "过滤", "萃取", "浓缩", "干燥"]
        details = parse_action_details(",".join(cycle * 10))
        self.assertEqual(len(details["actions"]), 80)
        self.assertEqual(details["loop_truncated"]["truncated_length"], 80)


if __name__ == "__main__":
    unittest.main()
