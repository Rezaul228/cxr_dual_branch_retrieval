#!/usr/bin/env python3
"""Cross-split near-duplicate audit for Indiana shards (original images only)."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


def base_case_id(study_id):
    s = str(study_id)
    if "_orig" in s:
        return s.split("_orig")[0]
    if "_aug_" in s:
        return s.split("_aug_")[0]
    return s


def is_original(study_id):
    return "_orig" in str(study_id)


def caption_key(tokens):
    return tuple(int(x) for x in tokens if int(x) != 0)


def caption_token_set(tokens):
    return frozenset(int(x) for x in tokens if int(x) != 0)


def min_shared_for_jaccard(size_a, size_b, threshold):
    return int(np.ceil(threshold * (size_a + size_b) / (1.0 + threshold)))


def load_indiana_reports(meta_csv):
    text_by_base = {}
    with open(meta_csv, newline="") as f:
        for row in csv.DictReader(f):
            bid = str(row["study_id"]).strip()
            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            labels = (row.get("labels") or "").strip()
            full = " ".join(x for x in [findings, impression] if x)
            norm = " ".join(full.lower().split())
            is_normal = labels.lower() == "normal" or bool(
                re.search(
                    r"\bno acute\b|\bnormal\b|\bclear lungs\b|\bno focal consolidation\b",
                    norm,
                )
            )
            text_by_base[bid] = {
                "norm_text": norm,
                "is_normal": is_normal,
                "labels": labels,
            }
    return text_by_base


def process_shard(args):
    path, split = args
    with open(path, "rb") as f:
        shard = pickle.load(f)
    rows = []
    for i in range(len(shard["study_ids"])):
        sid = str(shard["study_ids"][i])
        if not is_original(sid):
            continue
        base = base_case_id(sid)
        cap = shard["captions"][i]
        img = shard["images"][i]
        if img.max() <= 1.0:
            g = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        else:
            g = (
                0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            ) / 255.0
        vec = g[::8, ::8].flatten().astype(np.float32)
        rows.append(
            {
                "split": split,
                "study_id": sid,
                "base_id": base,
                "caption_exact": caption_key(cap),
                "caption_set": caption_token_set(cap),
                "caption_len": len(caption_token_set(cap)),
                "img_vec": vec,
            }
        )
    return rows


def jaccard_pairs_from_reps(reps, set_key, threshold, partial_only=False):
    inv = defaultdict(list)
    sets = []
    for idx, r in enumerate(reps):
        s = r[set_key]
        if not s:
            sets.append(None)
            continue
        sets.append(s)
        for tok in s:
            inv[tok].append(idx)

    pairs = []
    seen = set()
    for i, sa in enumerate(sets):
        if sa is None:
            continue
        candidates = Counter()
        for tok in sa:
            for j in inv[tok]:
                if j <= i:
                    continue
                candidates[j] += 1

        for j, shared in candidates.items():
            sb = sets[j]
            if sb is None:
                continue
            min_shared = min_shared_for_jaccard(len(sa), len(sb), threshold)
            if shared < min_shared:
                continue
            jac = shared / len(sa | sb)
            if jac < threshold:
                continue
            if partial_only and jac >= 1.0:
                continue

            a, b = reps[i], reps[j]
            if a["base_id"] == b["base_id"] or a["split"] == b["split"]:
                continue
            key = tuple(
                sorted([(a["split"], a["base_id"]), (b["split"], b["base_id"])])
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "split_a": a["split"],
                    "base_a": a["base_id"],
                    "study_a": a["study_id"],
                    "split_b": b["split"],
                    "base_b": b["base_id"],
                    "study_b": b["study_id"],
                    "jaccard": round(jac, 4),
                    "tokens_a": len(sa),
                    "tokens_b": len(sb),
                }
            )
    return pairs


def image_similarity_pairs(rep_list, threshold):
    buckets = defaultdict(list)
    for r in rep_list:
        buckets[int(r["img_vec"].mean() * 10)].append(r)

    pairs = []
    seen = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        vecs = np.stack([m["img_vec"] for m in members]).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs_n = vecs / norms
        sim = vecs_n @ vecs_n.T
        n = len(members)
        for i in range(n):
            for j in range(i + 1, n):
                corr = float(sim[i, j])
                if corr < threshold:
                    continue
                a, b = members[i], members[j]
                if a["base_id"] == b["base_id"] or a["split"] == b["split"]:
                    continue
                key = tuple(
                    sorted([(a["split"], a["base_id"]), (b["split"], b["base_id"])])
                )
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "split_a": a["split"],
                        "base_a": a["base_id"],
                        "study_a": a["study_id"],
                        "split_b": b["split"],
                        "base_b": b["base_id"],
                        "study_b": b["study_id"],
                        "cosine_similarity": round(corr, 4),
                    }
                )
    pairs.sort(key=lambda x: -x["cosine_similarity"])
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-split near-duplicate audit for Indiana CXR shards "
            "(original images only)."
        )
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Shard root with train/val/test/*.pkl",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for CSV reports",
    )
    parser.add_argument(
        "--meta-csv",
        required=True,
        help="Indiana study metadata CSV with findings/impression/labels",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.90,
        help="Jaccard threshold for near-duplicate detection (default: 0.90)",
    )
    parser.add_argument(
        "--img-threshold",
        type=float,
        default=0.985,
        help="Minimum cosine similarity for image pairs (default: 0.985)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for shard loading (default: min(8, cpu_count))",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.base_dir
    out_dir = args.out_dir
    jaccard_th = args.jaccard_threshold
    img_th = args.img_threshold
    workers = args.workers if args.workers is not None else min(8, cpu_count())

    os.makedirs(out_dir, exist_ok=True)
    text_by_base = load_indiana_reports(args.meta_csv)

    print("Loading original-only shards...", flush=True)
    shard_paths = [
        (p, s)
        for s in ["train", "val", "test"]
        for p in sorted(glob.glob(os.path.join(base_dir, s, "*.pkl")))
    ]
    if not shard_paths:
        raise FileNotFoundError(
            f"No shard pickles found under {base_dir}/{{train,val,test}}/"
        )

    all_rows = []
    with Pool(max(1, min(workers, cpu_count()))) as pool:
        for i, rows in enumerate(
            pool.imap_unordered(process_shard, shard_paths, chunksize=8), 1
        ):
            all_rows.extend(rows)
            if i % 100 == 0 or i == len(shard_paths):
                print(
                    f"  {i}/{len(shard_paths)} shards, {len(all_rows)} original samples",
                    flush=True,
                )
    print(f"Total original samples: {len(all_rows)}", flush=True)

    exact_cap_groups = defaultdict(list)
    for r in all_rows:
        if r["caption_exact"]:
            exact_cap_groups[r["caption_exact"]].append(r)

    exact_cross_pairs = []
    exact_cross_groups = []
    for cap, members in exact_cap_groups.items():
        bases = {m["base_id"] for m in members}
        splits = {m["split"] for m in members}
        if len(bases) < 2 or len(splits) < 2:
            continue
        exact_cross_groups.append(
            {
                "caption_token_len": len(cap),
                "num_samples": len(members),
                "num_base_cases": len(bases),
                "splits": ",".join(sorted(splits)),
                "base_ids": ",".join(sorted(bases)[:12])
                + ("..." if len(bases) > 12 else ""),
                "example_study_ids": ",".join(m["study_id"] for m in members[:6]),
            }
        )
        by_split_base = defaultdict(set)
        for m in members:
            by_split_base[(m["split"], m["base_id"])].add(m["study_id"])
        keys = list(by_split_base.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (s1, b1), (s2, b2) = keys[i], keys[j]
                if b1 == b2 or s1 == s2:
                    continue
                exact_cross_pairs.append((s1, b1, s2, b2, len(cap)))

    rep_list = all_rows
    print("Near-duplicate token scan...", flush=True)
    near_pairs = jaccard_pairs_from_reps(
        rep_list, "caption_set", threshold=jaccard_th
    )
    near_exact = [p for p in near_pairs if p["jaccard"] == 1.0]
    near_partial = [p for p in near_pairs if p["jaccard"] < 1.0]
    print(
        f"  near pairs: {len(near_pairs)} "
        f"({len(near_exact)} exact; {len(near_partial)} partial)",
        flush=True,
    )

    study_info = {r["study_id"]: r for r in all_rows}
    text_groups = defaultdict(list)
    for r in all_rows:
        info = text_by_base.get(r["base_id"], {"norm_text": "", "is_normal": False})
        if info["norm_text"]:
            text_groups[info["norm_text"]].append(r["study_id"])

    cross_text_groups = []
    for norm, sids in text_groups.items():
        if len(sids) < 2:
            continue
        bases = {study_info[s]["base_id"] for s in sids}
        if len(bases) < 2:
            continue
        split_presence = defaultdict(set)
        for s in sids:
            split_presence[study_info[s]["split"]].add(study_info[s]["base_id"])
        splits_present = [sp for sp, v in split_presence.items() if v]
        if len(splits_present) < 2:
            continue
        cross_text_groups.append(
            {
                "norm_text_preview": norm[:140] + ("..." if len(norm) > 140 else ""),
                "num_samples": len(sids),
                "num_base_cases": len(bases),
                "splits_spanned": ",".join(sorted(splits_present)),
                "num_normal_labels": sum(
                    1
                    for s in sids
                    if text_by_base.get(study_info[s]["base_id"], {}).get("is_normal")
                ),
                "text_len_chars": len(norm),
                "base_ids": ",".join(sorted(bases)[:12])
                + ("..." if len(bases) > 12 else ""),
            }
        )
    cross_text_groups.sort(key=lambda x: (-x["num_base_cases"], -x["text_len_chars"]))

    text_reps = []
    for r in all_rows:
        norm = text_by_base.get(r["base_id"], {}).get("norm_text", "")
        if not norm:
            continue
        text_reps.append(
            {
                "split": r["split"],
                "base_id": r["base_id"],
                "study_id": r["study_id"],
                "word_set": frozenset(norm.split()),
            }
        )
    partial_pairs = jaccard_pairs_from_reps(
        text_reps, "word_set", threshold=jaccard_th, partial_only=True
    )
    print(f"  partial text pairs: {len(partial_pairs)}", flush=True)

    print("Image similarity scan...", flush=True)
    img_pairs = image_similarity_pairs(rep_list, threshold=img_th)
    img995 = [p for p in img_pairs if p["cosine_similarity"] >= 0.995]
    img99 = [p for p in img_pairs if p["cosine_similarity"] >= 0.99]
    print(
        f"  image pairs >=0.995: {len(img995)}; "
        f">=0.99: {len(img99)}; >={img_th}: {len(img_pairs)}",
        flush=True,
    )

    all_bases = {r["base_id"] for r in all_rows}
    test_bases = {r["base_id"] for r in all_rows if r["split"] == "test"}
    n_samples = len(all_rows)
    n_bases = len(all_bases)

    base_splits = defaultdict(set)
    for r in all_rows:
        base_splits[r["base_id"]].add(r["split"])
    n_cross_split_bases = sum(1 for b, sp in base_splits.items() if len(sp) > 1)

    bases_near = set()
    train_linked_near = set()
    train_linked_exact = set()
    for p in near_pairs:
        bases_near.add(p["base_a"])
        bases_near.add(p["base_b"])
        if p["split_a"] == "test" and p["split_b"] == "train":
            train_linked_near.add(p["base_a"])
        if p["split_b"] == "test" and p["split_a"] == "train":
            train_linked_near.add(p["base_b"])
        if p["jaccard"] == 1.0:
            if p["split_a"] == "test" and p["split_b"] == "train":
                train_linked_exact.add(p["base_a"])
            if p["split_b"] == "test" and p["split_a"] == "train":
                train_linked_exact.add(p["base_b"])

    exact_split_counts = Counter(
        tuple(sorted((a, b))) for a, _, b, _, _ in exact_cross_pairs
    )
    near_split_counts = Counter(
        tuple(sorted((p["split_a"], p["split_b"]))) for p in near_pairs
    )

    pct_near = (100 * len(bases_near) / n_bases) if n_bases else 0.0
    pct_train_near = (
        (100 * len(train_linked_near) / len(test_bases)) if test_bases else 0.0
    )
    pct_train_exact = (
        (100 * len(train_linked_exact) / len(test_bases)) if test_bases else 0.0
    )
    n_all_normal = sum(
        1 for g in cross_text_groups if g["num_normal_labels"] == g["num_samples"]
    )
    n_span_all = sum(
        1
        for g in cross_text_groups
        if all(x in g["splits_spanned"] for x in ["train", "val", "test"])
    )

    summary = [
        [
            "Dataset scope",
            (
                f"{n_samples:,} original samples (_orig); {n_bases:,} base cases; "
                f"{n_cross_split_bases} cross-split case overlap"
            ),
            "—",
        ],
        [
            "A1",
            "Cross-split EXACT duplicate tokenized captions (different patients)",
            f"{len(exact_cross_groups)} groups; {len(exact_cross_pairs)} patient-pairs",
        ],
        [
            "A2",
            f"Cross-split NEAR-duplicate tokenized captions (Jaccard>={jaccard_th:.2f})",
            (
                f"{len(near_pairs)} pairs "
                f"({len(near_exact)} exact; {len(near_partial)} partial)"
            ),
        ],
        [
            "A3",
            "Unique base cases in near-duplicate caption pairs",
            f"{len(bases_near)} / {n_bases} ({pct_near:.1f}%)",
        ],
        [
            "A4",
            "Test cases with near-duplicate caption linked to TRAIN",
            f"{len(train_linked_near)} / {len(test_bases)} ({pct_train_near:.1f}%)",
        ],
        [
            "A5",
            "Test cases with EXACT duplicate caption linked to TRAIN",
            f"{len(train_linked_exact)} / {len(test_bases)} ({pct_train_exact:.1f}%)",
        ],
        [
            "B1",
            "Cross-split EXACT duplicate original report text (findings+impression)",
            (
                f"{len(cross_text_groups)} groups; "
                f"{sum(g['num_base_cases'] for g in cross_text_groups)} cases total"
            ),
        ],
        [
            "B2",
            "Exact-text groups spanning train+val+test",
            f"{n_span_all} groups",
        ],
        [
            "B3",
            "Exact-text groups where ALL cases labeled normal",
            f"{n_all_normal} / {len(cross_text_groups)}",
        ],
        [
            "B4",
            (
                f"Cross-split PARTIAL duplicate original text "
                f"(word Jaccard {jaccard_th:.2f}–<1.0)"
            ),
            f"{len(partial_pairs)} patient-pairs",
        ],
        [
            "C1",
            "Cross-split visually similar images (cosine>=0.995 on 28x28 gray)",
            f"{len(img995)} patient-pairs",
        ],
        [
            "C2",
            "Cross-split visually similar images (cosine>=0.99)",
            f"{len(img99)} patient-pairs",
        ],
        [
            "C3",
            f"Cross-split visually similar images (cosine>={img_th})",
            f"{len(img_pairs)} patient-pairs",
        ],
    ]

    pd.DataFrame(summary, columns=["ID", "Metric", "Result"]).to_csv(
        os.path.join(out_dir, "13_reviewer_master_table.csv"), index=False
    )
    pd.DataFrame(exact_cross_groups).to_csv(
        os.path.join(out_dir, "02_exact_duplicate_caption_groups.csv"), index=False
    )
    pd.DataFrame(
        [{"split_pair": f"{a}-{b}", "num_pairs": c} for (a, b), c in sorted(exact_split_counts.items())]
    ).to_csv(os.path.join(out_dir, "03_exact_duplicate_pairs_by_split.csv"), index=False)
    pd.DataFrame(near_pairs).to_csv(
        os.path.join(out_dir, "04_near_duplicate_caption_pairs.csv"), index=False
    )
    pd.DataFrame(
        [{"split_pair": f"{a}-{b}", "num_pairs": c} for (a, b), c in sorted(near_split_counts.items())]
    ).to_csv(os.path.join(out_dir, "05_near_duplicate_pairs_by_split.csv"), index=False)
    pd.DataFrame(cross_text_groups).to_csv(
        os.path.join(out_dir, "09_exact_duplicate_original_text_groups.csv"),
        index=False,
    )
    pd.DataFrame(partial_pairs).to_csv(
        os.path.join(out_dir, "11_partial_duplicate_original_text_pairs.csv"),
        index=False,
    )
    pd.DataFrame(img_pairs).to_csv(
        os.path.join(out_dir, "06_visually_similar_image_pairs.csv"), index=False
    )
    pd.DataFrame(
        [[row[1], row[2]] for row in summary[1:]], columns=["Metric", "Result"]
    ).to_csv(os.path.join(out_dir, "01_summary.csv"), index=False)
    if img995:
        pd.DataFrame(img995).to_csv(
            os.path.join(out_dir, "06b_visually_similar_image_pairs_ge_0.995.csv"),
            index=False,
        )

    print("\n=== INDIANA ORIG-ONLY REVIEWER SUMMARY ===")
    for row in summary:
        print(row)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
