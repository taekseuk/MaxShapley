#!/usr/bin/env python3
"""
Create perturbed MaxShapley dataset files.

Compatible with the dataset structure described in the MaxShapley repo:
- question
- context: list of [title, sentences] or {"title": ..., "sentences": ...}
- answer
- supporting_facts

Implements:
- duplicate relevant docs
- add zero-information docs
- add contradictory docs
- optionally shuffle context order

Usage:
  python perturb_maxshapley_dataset.py \
      --input data/musique_annotated_subset.json \
      --output data/musique_annotated_subset_perturbed.json \
      --duplicate-relevant 1 \
      --add-zero-docs 1 \
      --add-contradictory-docs 1 \
      --shuffle-context

"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_context_item(item: Any) -> Tuple[str, List[str], str]:
    """
    Returns (title, sentences, format_tag)
    format_tag is either 'list' or 'dict' so we can round-trip.
    """
    if isinstance(item, list) and len(item) == 2:
        title, sentences = item
        if isinstance(sentences, str):
            sentences = [sentences]
        return str(title), [str(s) for s in sentences], "list"
    if isinstance(item, dict):
        title = item.get("title", "")
        sentences = item.get("sentences", [])
        if isinstance(sentences, str):
            sentences = [sentences]
        return str(title), [str(s) for s in sentences], "dict"
    raise ValueError(f"Unsupported context item format: {type(item)} -> {item!r}")


def _pack_context_item(title: str, sentences: List[str], format_tag: str) -> Any:
    if format_tag == "list":
        return [title, sentences]
    return {"title": title, "sentences": sentences}


def _relevant_titles(sample: Dict[str, Any]) -> set[str]:
    rel = set()
    for sf in sample.get("supporting_facts", []):
        if isinstance(sf, list) and len(sf) >= 1:
            title = sf[0]
            score = sf[1] if len(sf) > 1 else None
            if isinstance(score, (int, float)):
                if score > 0:
                    rel.add(str(title))
            else:
                rel.add(str(title))
        elif isinstance(sf, tuple) and len(sf) >= 1:
            title = sf[0]
            score = sf[1] if len(sf) > 1 else None
            if isinstance(score, (int, float)):
                if score > 0:
                    rel.add(str(title))
            else:
                rel.add(str(title))
    return rel


def _append_supporting_fact(sample: Dict[str, Any], title: str) -> None:
    sfs = sample.setdefault("supporting_facts", [])
    if not sfs:
        sfs.append([title, 1])
        return
    template = sfs[0]
    if isinstance(template, list):
        if len(template) >= 2 and isinstance(template[1], (int, float)):
            sfs.append([title, 1])
        else:
            sfs.append([title, 0])
    else:
        sfs.append([title, 1])


def _pick_wrong_answer(sample: Dict[str, Any], context_titles: List[str], rng: random.Random) -> str:
    answer = str(sample.get("answer", "")).strip()
    candidates = []

    # Titles are often entities and make decent wrong-answer candidates.
    for t in context_titles:
        cleaned = re.sub(r"\s*\([^)]*\)\s*", "", t).strip()
        if cleaned and cleaned.lower() != answer.lower():
            candidates.append(cleaned)

    # Some generic fallbacks.
    fallbacks = [
        "an unrelated entity",
        "none of the listed sources",
        "a different person",
        "a different company",
        "a different location",
    ]
    candidates.extend(fallbacks)
    return rng.choice(candidates) if candidates else "an unrelated entity"


def _make_duplicate_doc(title: str, sentences: List[str], dup_idx: int) -> Tuple[str, List[str]]:
    return f"{title} [DUPLICATE {dup_idx}]", copy.deepcopy(sentences)


def _make_zero_doc(idx: int) -> Tuple[str, List[str]]:
    return (
        f"ZeroDoc_{idx}",
        [
            "N/A.",
            "This passage contains no useful information for answering the question.",
            "It is intentionally included as a zero-information distractor."
        ],
    )


def _make_contradictory_doc(question: str, answer: str, wrong_answer: str, idx: int) -> Tuple[str, List[str]]:
    # Deliberately fluent, on-topic, and wrong.
    title = f"ContradictoryDoc_{idx}"
    sentences = [
        f"This source provides a conflicting answer to the question: {question}",
        f"It claims that the correct answer is {wrong_answer}, not {answer}.",
        f"This document is intentionally adversarial and should not receive positive attribution if the method is robust."
    ]
    return title, sentences


def perturb_sample(
    sample: Dict[str, Any],
    duplicate_relevant: int,
    add_zero_docs: int,
    add_contradictory_docs: int,
    shuffle_context: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    out = copy.deepcopy(sample)
    original_context = out.get("context", [])
    if not isinstance(original_context, list):
        raise ValueError("sample['context'] must be a list")

    parsed = [_normalize_context_item(x) for x in original_context]
    rel_titles = _relevant_titles(out)
    relevant_items = [(t, s, f) for (t, s, f) in parsed if t in rel_titles]
    all_titles = [t for (t, _, _) in parsed]

    perturb_log = {
        "duplicate_relevant": [],
        "zero_docs": [],
        "contradictory_docs": [],
        "shuffled": bool(shuffle_context),
    }

    # Duplicate relevant docs and mark duplicates as relevant for evaluation.
    for i in range(duplicate_relevant):
        if not relevant_items:
            break
        t, s, f = relevant_items[i % len(relevant_items)]
        new_title, new_sentences = _make_duplicate_doc(t, s, i + 1)
        parsed.append((new_title, new_sentences, f))
        _append_supporting_fact(out, new_title)
        perturb_log["duplicate_relevant"].append(new_title)

    # Add explicit zero-information docs.
    for i in range(add_zero_docs):
        z_title, z_sentences = _make_zero_doc(i + 1)
        parsed.append((z_title, z_sentences, "list"))
        perturb_log["zero_docs"].append(z_title)

    # Add contradictory docs; do NOT mark them as supporting.
    q = str(out.get("question", ""))
    a = str(out.get("answer", "")).strip()
    for i in range(add_contradictory_docs):
        wrong = _pick_wrong_answer(out, all_titles, rng)
        c_title, c_sentences = _make_contradictory_doc(q, a, wrong, i + 1)
        parsed.append((c_title, c_sentences, "list"))
        perturb_log["contradictory_docs"].append({"title": c_title, "wrong_answer": wrong})

    if shuffle_context:
        rng.shuffle(parsed)

    out["context"] = [_pack_context_item(t, s, f) for (t, s, f) in parsed]
    out["perturbations"] = perturb_log
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to original dataset JSON")
    ap.add_argument("--output", required=True, help="Path to write perturbed dataset JSON")
    ap.add_argument("--duplicate-relevant", type=int, default=0)
    ap.add_argument("--add-zero-docs", type=int, default=0)
    ap.add_argument("--add-contradictory-docs", type=int, default=0)
    ap.add_argument("--shuffle-context", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="Only perturb first N samples")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Expected top-level dataset JSON to be a list of samples")

    out = []
    for idx, sample in enumerate(data):
        if args.limit is not None and idx >= args.limit:
            out.append(copy.deepcopy(sample))
            continue
        out.append(
            perturb_sample(
                sample=sample,
                duplicate_relevant=args.duplicate_relevant,
                add_zero_docs=args.add_zero_docs,
                add_contradictory_docs=args.add_contradictory_docs,
                shuffle_context=args.shuffle_context,
                rng=rng,
            )
        )

    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} samples to {out_path}")


if __name__ == "__main__":
    main()
