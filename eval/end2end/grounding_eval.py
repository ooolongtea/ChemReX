"""Core grounding evaluation implementing all 11 metric categories.

Metrics:
  3. Text identifier recognition (identifier-set P/R/F1)
  4. Figure label OCR (IoU≥0.5 matching, label exact match, CER)
  5. Molecule box detection (IoU≥0.5 P/R/F1)
  6. OCSR (valid rate, strict/relaxed canonical exact match)
  7. Label-to-molecule linking (P/R/F1)
  8. Identifier-to-label linking (P/R/F1)
  9. Identifier-to-SMILES grounding — MAIN (P/R/F1 + relaxed F1)
  10. End-to-end grounding accuracy
  11. Error analysis (per failed gold identifier)
"""

from typing import Dict, List, Optional, Set, Tuple

from eval.end2end.normalize import (
    canonicalize_smiles,
    cer,
    greedy_iou_match,
    iou,
    is_valid_smiles,
    normalize_identifier,
)


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def evaluate_sample(gold: dict, pred: dict) -> dict:
    """Evaluate one sample. Returns metrics dict + error_cases list."""

    errors: List[dict] = []
    gold_idx = gold["openexp_index"]

    # ── Metric 3: Text identifier recognition ──
    gold_ids: Set[str] = {
        normalize_identifier(m["identifier"]) for m in gold["text_mentions"]
    }
    pred_ids: Set[str] = {
        normalize_identifier(m["identifier"]) for m in pred["text_mentions"]
    }
    text_tp = len(gold_ids & pred_ids)
    text_fp = len(pred_ids - gold_ids)
    text_fn = len(gold_ids - pred_ids)
    text_p, text_r, text_f1 = _prf(text_tp, text_fp, text_fn)

    # Track spurious text ids
    for spurious_id in pred_ids - gold_ids:
        errors.append(
            {
                "openexp_index": gold_idx,
                "gold_identifier": None,
                "error_type": "TEXT_ID_SPURIOUS",
                "details": {"pred_identifier": spurious_id},
            }
        )

    # ── Metric 4: Figure label OCR ──
    pred_label_boxes = [fl["bbox"] for fl in pred["figure_labels"]]
    gold_label_boxes = [fl["bbox"] for fl in gold["figure_labels"]]
    label_matches = greedy_iou_match(
        pred_label_boxes, gold_label_boxes, 0.5, center_fallback=True
    )

    label_tp = len(label_matches)
    label_fp = len(pred["figure_labels"]) - label_tp
    label_fn = len(gold["figure_labels"]) - label_tp
    label_det_p, label_det_r, label_det_f1 = _prf(label_tp, label_fp, label_fn)

    label_exact = 0
    cer_sum = 0.0
    for pi, gi, _ in label_matches:
        pl = pred["figure_labels"][pi]["label"]
        gl = gold["figure_labels"][gi]["label"]
        if normalize_identifier(pl) == normalize_identifier(gl):
            label_exact += 1
        cer_sum += cer(pl, gl)
    label_ocr_em = label_exact / len(label_matches) if label_matches else 0.0
    label_cer_val = cer_sum / len(label_matches) if label_matches else 0.0

    # ── Metric 5: Molecule box detection ──
    pred_mol_boxes = [mb["bbox"] for mb in pred["molecule_boxes"]]
    gold_mol_boxes = [mb["bbox"] for mb in gold["molecule_boxes"]]
    mol_matches = greedy_iou_match(pred_mol_boxes, gold_mol_boxes, 0.5)

    mol_tp = len(mol_matches)
    mol_fp = len(pred["molecule_boxes"]) - mol_tp
    mol_fn = len(gold["molecule_boxes"]) - mol_tp
    molbox_p, molbox_r, molbox_f1 = _prf(mol_tp, mol_fp, mol_fn)

    # ── Metric 6: OCSR ──
    ocsr_valid = 0
    ocsr_strict = 0
    ocsr_relaxed = 0
    for pi, gi, _ in mol_matches:
        psmi = pred["molecule_boxes"][pi]["smiles"]
        gsmi = gold["molecule_boxes"][gi]["smiles"]
        if is_valid_smiles(psmi):
            ocsr_valid += 1
            pc_strict = canonicalize_smiles(psmi, isomeric=True)
            gc_strict = canonicalize_smiles(gsmi, isomeric=True)
            if pc_strict and gc_strict and pc_strict == gc_strict:
                ocsr_strict += 1
            pc_relaxed = canonicalize_smiles(psmi, isomeric=False)
            gc_relaxed = canonicalize_smiles(gsmi, isomeric=False)
            if pc_relaxed and gc_relaxed and pc_relaxed == gc_relaxed:
                ocsr_relaxed += 1
    n_matched_mols = len(mol_matches)
    ocsr_valid_rate = ocsr_valid / n_matched_mols if n_matched_mols else 0.0
    ocsr_strict_exact = ocsr_strict / n_matched_mols if n_matched_mols else 0.0
    ocsr_relaxed_exact = ocsr_relaxed / n_matched_mols if n_matched_mols else 0.0

    # ── Mappings for link & E2E evaluation ──
    pred2gold_label = {pi: gi for pi, gi, _ in label_matches}
    pred2gold_mol = {pi: gi for pi, gi, _ in mol_matches}
    gold2pred_label = {gi: pi for pi, gi, _ in label_matches}
    gold2pred_mol = {gi: pi for pi, gi, _ in mol_matches}

    # ── Metric 7: Label→molecule linking ──
    gold_link_set: Set[Tuple[int, int]] = {
        (l["label_idx"], l["mol_idx"]) for l in gold["label_mol_links"]
    }
    pred_links: List[Tuple[int, int]] = [
        (l["label_idx"], l["mol_idx"]) for l in pred["label_mol_links"]
    ]
    link_tp = 0
    for pl_label, pl_mol in pred_links:
        if pl_label in pred2gold_label and pl_mol in pred2gold_mol:
            gl = pred2gold_label[pl_label]
            gm = pred2gold_mol[pl_mol]
            if (gl, gm) in gold_link_set:
                link_tp += 1
    link_fp = len(pred_links) - link_tp
    link_fn = len(gold_link_set) - link_tp
    link_p, link_r, link_f1 = _prf(link_tp, link_fp, link_fn)

    # ── Metric 8: Identifier→label linking ──
    pred_id_label: Set[Tuple[str, str]] = set()
    for tm in pred["text_mentions"]:
        nid = normalize_identifier(tm["identifier"])
        for fl in pred["figure_labels"]:
            nlabel = normalize_identifier(fl["label"])
            if nid == nlabel:
                pred_id_label.add((nid, nlabel))

    gold_id_label: Set[Tuple[str, str]] = set()
    for il in gold["id_label_links"]:
        nid = normalize_identifier(il["identifier"])
        li = il["label_idx"]
        if li < len(gold["figure_labels"]):
            nlabel = normalize_identifier(gold["figure_labels"][li]["label"])
            gold_id_label.add((nid, nlabel))

    id_label_tp = len(pred_id_label & gold_id_label)
    id_label_fp = len(pred_id_label - gold_id_label)
    id_label_fn = len(gold_id_label - pred_id_label)
    id_label_p, id_label_r, id_label_f1 = _prf(id_label_tp, id_label_fp, id_label_fn)

    # ── Metric 9: ID→SMILES grounding (MAIN) ──
    pred_ground: Dict[str, str] = {
        normalize_identifier(k): v for k, v in pred["id_smiles_mapping"].items()
    }
    gold_ground: Dict[str, str] = {
        normalize_identifier(g["identifier"]): g["smiles"] for g in gold["grounding"]
    }
    grounding_tp = 0
    grounding_tp_relaxed = 0
    for nid, psmi in pred_ground.items():
        gsmi = gold_ground.get(nid)
        if gsmi is None:
            continue
        if not is_valid_smiles(psmi):
            continue
        pc = canonicalize_smiles(psmi, isomeric=True)
        gc = canonicalize_smiles(gsmi, isomeric=True)
        if pc and gc and pc == gc:
            grounding_tp += 1
        pcr = canonicalize_smiles(psmi, isomeric=False)
        gcr = canonicalize_smiles(gsmi, isomeric=False)
        if pcr and gcr and pcr == gcr:
            grounding_tp_relaxed += 1
    grounding_fp = len(pred_ground) - grounding_tp
    grounding_fn = len(gold_ground) - grounding_tp
    grounding_p, grounding_r, grounding_f1 = _prf(grounding_tp, grounding_fp, grounding_fn)
    _, _, grounding_relaxed_f1 = _prf(grounding_tp_relaxed, grounding_fp, grounding_fn)

    # ── Metric 10 & 11: E2E accuracy + error analysis ──
    # Build gold id → (label_idx, mol_idx, smiles) mapping
    gold_id_to_label: Dict[str, int] = {}
    for il in gold["id_label_links"]:
        gold_id_to_label[normalize_identifier(il["identifier"])] = il["label_idx"]

    gold_label_to_mol: Dict[int, int] = {}
    for lm in gold["label_mol_links"]:
        gold_label_to_mol[lm["label_idx"]] = lm["mol_idx"]

    e2e_correct = 0
    total_gold_ids = 0

    for g in gold["grounding"]:
        nid = normalize_identifier(g["identifier"])
        gold_smi = g["smiles"]
        total_gold_ids += 1

        # Step 1: text id recognized?
        if nid not in pred_ids:
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "TEXT_ID_MISS",
                    "details": {},
                }
            )
            continue

        # Step 2: label detected (IoU ≥ 0.5)?
        g_label_idx = gold_id_to_label.get(nid)
        if g_label_idx is None or g_label_idx not in gold2pred_label:
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "LABEL_MISS",
                    "details": {},
                }
            )
            continue
        p_label_idx = gold2pred_label[g_label_idx]

        # Step 3: label OCR correct?
        pl = pred["figure_labels"][p_label_idx]["label"]
        gl = gold["figure_labels"][g_label_idx]["label"]
        if normalize_identifier(pl) != normalize_identifier(gl):
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "LABEL_OCR_ERROR",
                    "details": {"pred_label": pl, "gold_label": gl},
                }
            )
            continue

        # Step 4: mol box detected (IoU ≥ 0.5)?
        g_mol_idx = gold_label_to_mol.get(g_label_idx)
        if g_mol_idx is None or g_mol_idx not in gold2pred_mol:
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "MOLBOX_MISS",
                    "details": {},
                }
            )
            continue
        p_mol_idx = gold2pred_mol[g_mol_idx]

        # Step 5: link correct?
        has_link = any(
            pl == p_label_idx and pm == p_mol_idx for pl, pm in pred_links
        )
        if not has_link:
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "LINK_ERROR",
                    "details": {
                        "pred_label_idx": p_label_idx,
                        "pred_mol_idx": p_mol_idx,
                    },
                }
            )
            continue

        # Step 6: SMILES valid?
        pred_smi = pred["molecule_boxes"][p_mol_idx]["smiles"]
        if not is_valid_smiles(pred_smi):
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "OCSR_INVALID",
                    "details": {"pred_smiles": pred_smi},
                }
            )
            continue

        # Step 7: SMILES match?
        pc = canonicalize_smiles(pred_smi, isomeric=True)
        gc = canonicalize_smiles(gold_smi, isomeric=True)
        if not (pc and gc and pc == gc):
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "OCSR_MISMATCH",
                    "details": {
                        "pred_smiles": pred_smi,
                        "gold_smiles": gold_smi,
                        "canonical_pred": pc,
                        "canonical_gold": gc,
                    },
                }
            )
            continue

        # E2E chain passed — check M3 grounding
        pred_ground_smi = pred_ground.get(nid)
        if pred_ground_smi is None:
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "GROUNDING_MISS",
                    "details": {},
                }
            )
            continue

        if not is_valid_smiles(pred_ground_smi):
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "GROUNDING_WRONG",
                    "details": {"pred_smiles": pred_ground_smi, "gold_smiles": gold_smi},
                }
            )
            continue

        pgc = canonicalize_smiles(pred_ground_smi, isomeric=True)
        if not (pgc and gc and pgc == gc):
            errors.append(
                {
                    "openexp_index": gold_idx,
                    "gold_identifier": g["identifier"],
                    "error_type": "GROUNDING_WRONG",
                    "details": {
                        "pred_smiles": pred_ground_smi,
                        "gold_smiles": gold_smi,
                        "canonical_pred": pgc,
                        "canonical_gold": gc,
                    },
                }
            )
            continue

        e2e_correct += 1

    e2e_acc = e2e_correct / total_gold_ids if total_gold_ids > 0 else 0.0

    return {
        "openexp_index": gold_idx,
        "n_gold_identifiers": total_gold_ids,
        "text_id_tp": text_tp,
        "text_id_fp": text_fp,
        "text_id_fn": text_fn,
        "text_id_p": text_p,
        "text_id_r": text_r,
        "text_id_f1": text_f1,
        "label_det_tp": label_tp,
        "label_det_fp": label_fp,
        "label_det_fn": label_fn,
        "label_det_p": label_det_p,
        "label_det_r": label_det_r,
        "label_det_f1": label_det_f1,
        "label_ocr_em": label_ocr_em,
        "label_cer": label_cer_val,
        "molbox_tp": mol_tp,
        "molbox_fp": mol_fp,
        "molbox_fn": mol_fn,
        "molbox_p": molbox_p,
        "molbox_r": molbox_r,
        "molbox_f1": molbox_f1,
        "n_matched_mols": n_matched_mols,
        "ocsr_valid": ocsr_valid,
        "ocsr_strict": ocsr_strict,
        "ocsr_relaxed": ocsr_relaxed,
        "ocsr_valid_rate": ocsr_valid_rate,
        "ocsr_strict_exact": ocsr_strict_exact,
        "ocsr_relaxed_exact": ocsr_relaxed_exact,
        "link_tp": link_tp,
        "link_fp": link_fp,
        "link_fn": link_fn,
        "link_p": link_p,
        "link_r": link_r,
        "link_f1": link_f1,
        "id_label_tp": id_label_tp,
        "id_label_fp": id_label_fp,
        "id_label_fn": id_label_fn,
        "id_label_p": id_label_p,
        "id_label_r": id_label_r,
        "id_label_f1": id_label_f1,
        "grounding_tp": grounding_tp,
        "grounding_tp_relaxed": grounding_tp_relaxed,
        "grounding_fp": grounding_fp,
        "grounding_fn": grounding_fn,
        "grounding_p": grounding_p,
        "grounding_r": grounding_r,
        "grounding_f1": grounding_f1,
        "grounding_relaxed_f1": grounding_relaxed_f1,
        "e2e_correct": e2e_correct,
        "e2e_acc": e2e_acc,
        "errors": errors,
    }


def aggregate_results(per_sample: List[dict], dataset: str) -> dict:
    """Micro-average P/R/F1, macro-average rates, return summary dict."""

    n = len(per_sample)
    n_gold = sum(s["n_gold_identifiers"] for s in per_sample)

    # Micro P/R/F1
    def micro(tp_key, fp_key, fn_key):
        tp = sum(s[tp_key] for s in per_sample)
        fp = sum(s[fp_key] for s in per_sample)
        fn = sum(s[fn_key] for s in per_sample)
        return _prf(tp, fp, fn)

    text_p, text_r, text_f1 = micro("text_id_tp", "text_id_fp", "text_id_fn")
    label_p, label_r, label_f1 = micro("label_det_tp", "label_det_fp", "label_det_fn")
    mol_p, mol_r, mol_f1 = micro("molbox_tp", "molbox_fp", "molbox_fn")
    link_p, link_r, link_f1 = micro("link_tp", "link_fp", "link_fn")
    id_label_p, id_label_r, id_label_f1 = micro("id_label_tp", "id_label_fp", "id_label_fn")
    g_p, g_r, g_f1 = micro("grounding_tp", "grounding_fp", "grounding_fn")

    # Relaxed grounding F1 (micro-averaged)
    g_tp_relaxed = sum(s["grounding_tp_relaxed"] for s in per_sample)
    g_fp = sum(s["grounding_fp"] for s in per_sample)
    g_fn = sum(s["grounding_fn"] for s in per_sample)
    _, _, grounding_relaxed_f1 = _prf(g_tp_relaxed, g_fp, g_fn)

    # OCSR: macro-average over samples that have matched mols
    samples_with_mols = [s for s in per_sample if s["n_matched_mols"] > 0]
    ocsr_valid_rate = (
        sum(s["ocsr_valid_rate"] for s in samples_with_mols) / len(samples_with_mols)
        if samples_with_mols
        else 0.0
    )
    ocsr_strict_exact = (
        sum(s["ocsr_strict_exact"] for s in samples_with_mols) / len(samples_with_mols)
        if samples_with_mols
        else 0.0
    )
    ocsr_relaxed_exact = (
        sum(s["ocsr_relaxed_exact"] for s in samples_with_mols) / len(samples_with_mols)
        if samples_with_mols
        else 0.0
    )

    # Label OCR EM & CER: macro-average over samples with matched labels
    samples_with_labels = [s for s in per_sample if s["label_det_tp"] > 0]
    label_ocr_em = (
        sum(s["label_ocr_em"] for s in samples_with_labels) / len(samples_with_labels)
        if samples_with_labels
        else 0.0
    )
    label_cer = (
        sum(s["label_cer"] for s in samples_with_labels) / len(samples_with_labels)
        if samples_with_labels
        else 0.0
    )

    # E2E accuracy: micro
    e2e_correct = sum(s["e2e_correct"] for s in per_sample)
    e2e_acc = e2e_correct / n_gold if n_gold > 0 else 0.0

    return {
        "dataset": dataset,
        "n_samples": n,
        "n_gold_identifiers": n_gold,
        "text_id_p": f"{text_p:.4f}",
        "text_id_r": f"{text_r:.4f}",
        "text_id_f1": f"{text_f1:.4f}",
        "label_det_p": f"{label_p:.4f}",
        "label_det_r": f"{label_r:.4f}",
        "label_det_f1": f"{label_f1:.4f}",
        "label_ocr_em": f"{label_ocr_em:.4f}",
        "label_cer": f"{label_cer:.4f}",
        "molbox_p": f"{mol_p:.4f}",
        "molbox_r": f"{mol_r:.4f}",
        "molbox_f1": f"{mol_f1:.4f}",
        "ocsr_valid_rate": f"{ocsr_valid_rate:.4f}",
        "ocsr_strict_exact": f"{ocsr_strict_exact:.4f}",
        "ocsr_relaxed_exact": f"{ocsr_relaxed_exact:.4f}",
        "link_p": f"{link_p:.4f}",
        "link_r": f"{link_r:.4f}",
        "link_f1": f"{link_f1:.4f}",
        "grounding_p": f"{g_p:.4f}",
        "grounding_r": f"{g_r:.4f}",
        "grounding_f1": f"{g_f1:.4f}",
        "grounding_relaxed_f1": f"{grounding_relaxed_f1:.4f}",
        "e2e_acc": f"{e2e_acc:.4f}",
    }
