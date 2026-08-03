#!/usr/bin/env python3
"""Cross-split near-duplicate audit for MIMIC shard datasets."""

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

# Set in main() before worker pools that need report-file resolution.
REPORTS_ROOT = None


def caption_key(tokens):
    return tuple(int(x) for x in tokens if int(x) != 0)


def caption_token_set(tokens):
    return frozenset(int(x) for x in tokens if int(x) != 0)


def min_shared_for_jaccard(size_a, size_b, threshold):
    return int(np.ceil(threshold * (size_a + size_b) / (1.0 + threshold)))


def build_study_maps(meta_csv, filtered_csv):
    study_to_subject = {}
    with open(meta_csv, newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row["study_id"]).strip()
            if sid not in study_to_subject:
                study_to_subject[sid] = str(row["subject_id"]).strip()

    study_to_report = {}
    with open(filtered_csv, newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row["study_id"]).strip()
            rf = row.get("report_file", "")
            if sid not in study_to_report and rf:
                study_to_report[sid] = rf
    return study_to_subject, study_to_report


def read_report_text(rf, reports_root=None):
    root = reports_root if reports_root is not None else REPORTS_ROOT
    path = os.path.join(root, rf) if not os.path.isabs(rf) else rf
    if not os.path.exists(path):
        return "", False
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return "", False

    low = txt.lower()
    findings = ""
    impression = ""
    if "findings:" in low:
        part = (
            txt.split("FINDINGS:", 1)[-1]
            if "FINDINGS:" in txt
            else txt.split("findings:", 1)[-1]
        )
        impression_split = re.split(
            r"\n\s*IMPRESSION:|\n\s*impression:", part, maxsplit=1
        )
        findings = impression_split[0]
        if len(impression_split) > 1:
            impression = impression_split[1]
    else:
        findings = txt

    full = " ".join(x.strip() for x in [findings, impression] if x and x.strip())
    norm = " ".join(full.lower().split())
    is_normal = bool(
        re.search(
            r"\bno acute\b|\bnormal\b|\bclear lungs\b|\bno focal consolidation\b",
            norm,
        )
    )
    return norm, is_normal


def process_shard(args):
    path, split, study_to_subject = args
    with open(path, "rb") as f:
        shard = pickle.load(f)
    rows = []
    for i in range(len(shard["study_ids"])):
        sid = str(shard["study_ids"][i])
        subj = study_to_subject.get(sid, "UNKNOWN")
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
                "subject_id": subj,
                "caption_exact": caption_key(cap),
                "caption_set": caption_token_set(cap),
                "caption_len": len(caption_token_set(cap)),
                "img_vec": vec,
            }
        )
    return rows


def load_one_report(item):
    sid, rf = item
    if not rf:
        return sid, "", False
    norm, is_normal = read_report_text(rf)
    return sid, norm, is_normal


def load_reports_parallel(study_ids, study_to_report, workers):
    items = [(sid, study_to_report.get(sid)) for sid in study_ids]
    out = {}
    n_workers = max(1, min(workers, cpu_count(), len(items) or 1))
    with Pool(n_workers) as pool:
        for i, (sid, norm, is_normal) in enumerate(
            pool.imap_unordered(load_one_report, items, chunksize=256), 1
        ):
            out[sid] = {"norm_text": norm, "is_normal": is_normal}
            if i % 25000 == 0 or i == len(items):
                print(f"  reports {i}/{len(items)}", flush=True)
    return out


def jaccard_pairs_from_reps(reps, set_key, threshold, partial_only=False):
    """Find cross-split cross-patient pairs with Jaccard >= threshold."""
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
            if not partial_only and jac < 1.0 and threshold >= 1.0:
                continue

            a, b = reps[i], reps[j]
            if a["subject_id"] == b["subject_id"] or a["split"] == b["split"]:
                continue
            key = tuple(
                sorted([(a["split"], a["subject_id"]), (b["split"], b["subject_id"])])
            )
            if key in seen:
                continue
            seen.add(key)
            pairs.append(
                {
                    "split_a": a["split"],
                    "subject_a": a["subject_id"],
                    "study_a": a["study_id"],
                    "split_b": b["split"],
                    "subject_b": b["subject_id"],
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
                if a["subject_id"] == b["subject_id"] or a["split"] == b["split"]:
                    continue
                key = tuple(
                    sorted(
                        [(a["split"], a["subject_id"]), (b["split"], b["subject_id"])]
                    )
                )
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(
                    {
                        "split_a": a["split"],
                        "subject_a": a["subject_id"],
                        "study_a": a["study_id"],
                        "split_b": b["split"],
                        "subject_b": b["subject_id"],
                        "study_b": b["study_id"],
                        "cosine_similarity": round(corr, 4),
                    }
                )
    pairs.sort(key=lambda x: -x["cosine_similarity"])
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-split near-duplicate audit for MIMIC CXR shards."
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
        help="MIMIC metadata CSV with study_id and subject_id",
    )
    parser.add_argument(
        "--filtered-csv",
        required=True,
        help="Filtered metadata CSV with study_id and report_file",
    )
    parser.add_argument(
        "--reports-root",
        required=True,
        help="Root directory for MIMIC report text files",
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
        help="Parallel workers (default: min(8, cpu_count) for shards; "
        "min(16, cpu_count) for reports)",
    )
    return parser.parse_args()


def main():
    global REPORTS_ROOT
    args = parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir
    jaccard_th = args.jaccard_threshold
    img_th = args.img_threshold
    REPORTS_ROOT = args.reports_root

    shard_workers = args.workers if args.workers is not None else min(8, cpu_count())
    report_workers = args.workers if args.workers is not None else min(16, cpu_count())

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "_checkpoint_after_captions.pkl")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}...", flush=True)
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        all_rows = ckpt["all_rows"]
        exact_cross_groups = ckpt["exact_cross_groups"]
        exact_cross_pairs = ckpt["exact_cross_pairs"]
        rep_list = ckpt["rep_list"]
        near_pairs = ckpt["near_pairs"]
        near_exact = ckpt["near_exact"]
        near_partial = ckpt["near_partial"]
        study_to_report = ckpt["study_to_report"]
        print(
            f"  restored {len(all_rows)} samples, {len(near_pairs)} near pairs",
            flush=True,
        )
    else:
        print("Building study maps...", flush=True)
        study_to_subject, study_to_report = build_study_maps(
            args.meta_csv, args.filtered_csv
        )

        print("Loading shards...", flush=True)
        shard_paths = [
            (p, s, study_to_subject)
            for s in ["train", "val", "test"]
            for p in sorted(glob.glob(os.path.join(base_dir, s, "*.pkl")))
        ]
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard pickles found under {base_dir}/{{train,val,test}}/"
            )

        all_rows = []
        with Pool(max(1, min(shard_workers, cpu_count()))) as pool:
            for i, rows in enumerate(
                pool.imap_unordered(process_shard, shard_paths, chunksize=8), 1
            ):
                all_rows.extend(rows)
                if i % 200 == 0 or i == len(shard_paths):
                    print(
                        f"  {i}/{len(shard_paths)} shards, {len(all_rows)} samples",
                        flush=True,
                    )
        print(f"Total samples: {len(all_rows)}", flush=True)

        exact_cap_groups = defaultdict(list)
        for r in all_rows:
            if r["caption_exact"]:
                exact_cap_groups[r["caption_exact"]].append(r)

        exact_cross_pairs = []
        exact_cross_groups = []
        for cap, members in exact_cap_groups.items():
            bases = {m["subject_id"] for m in members}
            splits = {m["split"] for m in members}
            if len(bases) < 2 or len(splits) < 2:
                continue
            exact_cross_groups.append(
                {
                    "caption_token_len": len(cap),
                    "num_samples": len(members),
                    "num_patients": len(bases),
                    "splits": ",".join(sorted(splits)),
                    "subject_ids": ",".join(sorted(bases)[:12])
                    + ("..." if len(bases) > 12 else ""),
                    "example_study_ids": ",".join(m["study_id"] for m in members[:6]),
                }
            )
            by_split_subj = defaultdict(set)
            for m in members:
                by_split_subj[(m["split"], m["subject_id"])].add(m["study_id"])
            keys = list(by_split_subj.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    (s1, b1), (s2, b2) = keys[i], keys[j]
                    if b1 == b2 or s1 == s2:
                        continue
                    exact_cross_pairs.append((s1, b1, s2, b2, len(cap)))

        reps = {}
        for r in all_rows:
            key = (r["split"], r["subject_id"])
            if key not in reps:
                reps[key] = r
        rep_list = list(reps.values())
        print(f"Unique (split, patient) reps: {len(rep_list)}", flush=True)

        print("Near-duplicate token scan (inverted index)...", flush=True)
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

        print(f"Saving checkpoint {checkpoint_path}...", flush=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump(
                {
                    "all_rows": all_rows,
                    "exact_cross_groups": exact_cross_groups,
                    "exact_cross_pairs": exact_cross_pairs,
                    "rep_list": rep_list,
                    "near_pairs": near_pairs,
                    "near_exact": near_exact,
                    "near_partial": near_partial,
                    "study_to_report": study_to_report,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    print("Loading original report text...", flush=True)
    unique_studies = {r["study_id"] for r in all_rows}
    text_ckpt = os.path.join(out_dir, "_checkpoint_after_text.pkl")
    if os.path.exists(text_ckpt):
        print(f"Loading text checkpoint {text_ckpt}...", flush=True)
        with open(text_ckpt, "rb") as f:
            text_ckpt_data = pickle.load(f)
        text_by_study = text_ckpt_data["text_by_study"]
        cross_text_groups = text_ckpt_data["cross_text_groups"]
        partial_pairs = text_ckpt_data["partial_pairs"]
        print(
            f"  restored {len(text_by_study)} reports, "
            f"{len(partial_pairs)} partial pairs",
            flush=True,
        )
    else:
        text_by_study = load_reports_parallel(
            unique_studies, study_to_report, workers=report_workers
        )
        study_info = {r["study_id"]: r for r in all_rows}

        text_groups = defaultdict(list)
        for sid, info in text_by_study.items():
            if info["norm_text"]:
                text_groups[info["norm_text"]].append(sid)

        cross_text_groups = []
        for norm, sids in text_groups.items():
            if len(sids) < 2:
                continue
            patients = {study_info[s]["subject_id"] for s in sids if s in study_info}
            if len(patients) < 2:
                continue
            split_presence = defaultdict(set)
            for s in sids:
                if s not in study_info:
                    continue
                split_presence[study_info[s]["split"]].add(study_info[s]["subject_id"])
            splits_present = [sp for sp, v in split_presence.items() if v]
            if len(splits_present) < 2:
                continue
            cross_text_groups.append(
                {
                    "norm_text_preview": norm[:140] + ("..." if len(norm) > 140 else ""),
                    "num_samples": len(sids),
                    "num_patients": len(patients),
                    "splits_spanned": ",".join(sorted(splits_present)),
                    "num_normal_heuristic": sum(
                        1 for s in sids if text_by_study[s]["is_normal"]
                    ),
                    "text_len_chars": len(norm),
                    "subject_ids": ",".join(sorted(patients)[:12])
                    + ("..." if len(patients) > 12 else ""),
                }
            )
        cross_text_groups.sort(key=lambda x: (-x["num_patients"], -x["text_len_chars"]))

        print("Partial duplicate original text scan...", flush=True)
        text_reps = []
        for sid in unique_studies:
            norm = text_by_study[sid]["norm_text"]
            if not norm:
                continue
            r = study_info[sid]
            text_reps.append(
                {
                    "split": r["split"],
                    "subject_id": r["subject_id"],
                    "study_id": sid,
                    "word_set": frozenset(norm.split()),
                }
            )
        text_rep_map = {}
        for r in text_reps:
            key = (r["split"], r["subject_id"])
            if key not in text_rep_map:
                text_rep_map[key] = r
        text_rep_list = list(text_rep_map.values())
        partial_pairs = jaccard_pairs_from_reps(
            text_rep_list, "word_set", threshold=jaccard_th, partial_only=True
        )
        print(f"  partial text pairs: {len(partial_pairs)}", flush=True)

        with open(text_ckpt, "wb") as f:
            pickle.dump(
                {
                    "text_by_study": text_by_study,
                    "cross_text_groups": cross_text_groups,
                    "partial_pairs": partial_pairs,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    print("Image similarity scan...", flush=True)
    img_pairs = image_similarity_pairs(rep_list, threshold=img_th)
    img995 = [p for p in img_pairs if p["cosine_similarity"] >= 0.995]
    img99 = [p for p in img_pairs if p["cosine_similarity"] >= 0.99]
    print(
        f"  image pairs >=0.995: {len(img995)}; "
        f">=0.99: {len(img99)}; >={img_th}: {len(img_pairs)}",
        flush=True,
    )

    all_subjects = {r["subject_id"] for r in all_rows}
    test_pts = {r["subject_id"] for r in all_rows if r["split"] == "test"}
    n_samples = len(all_rows)
    n_patients = len(all_subjects)

    # Cross-split patient overlap (same subject_id in multiple splits)
    subject_splits = defaultdict(set)
    for r in all_rows:
        subject_splits[r["subject_id"]].add(r["split"])
    n_cross_split_patients = sum(1 for s, sp in subject_splits.items() if len(sp) > 1)

    subjects_near = set()
    train_linked_near = set()
    train_linked_exact = set()
    for p in near_pairs:
        subjects_near.add(p["subject_a"])
        subjects_near.add(p["subject_b"])
        if p["split_a"] == "test" and p["split_b"] == "train":
            train_linked_near.add(p["subject_a"])
        if p["split_b"] == "test" and p["split_a"] == "train":
            train_linked_near.add(p["subject_b"])
        if p["jaccard"] == 1.0:
            if p["split_a"] == "test" and p["split_b"] == "train":
                train_linked_exact.add(p["subject_a"])
            if p["split_b"] == "test" and p["split_a"] == "train":
                train_linked_exact.add(p["subject_b"])

    exact_split_counts = Counter(
        tuple(sorted((a, b))) for a, _, b, _, _ in exact_cross_pairs
    )
    near_split_counts = Counter(
        tuple(sorted((p["split_a"], p["split_b"]))) for p in near_pairs
    )

    pct_near = (100 * len(subjects_near) / n_patients) if n_patients else 0.0
    pct_train_near = (
        (100 * len(train_linked_near) / len(test_pts)) if test_pts else 0.0
    )
    pct_train_exact = (
        (100 * len(train_linked_exact) / len(test_pts)) if test_pts else 0.0
    )
    n_all_normal = sum(
        1
        for g in cross_text_groups
        if g["num_normal_heuristic"] == g["num_samples"]
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
                f"{n_samples:,} samples; {n_patients:,} patients; "
                f"{n_cross_split_patients} cross-split patient overlap"
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
            "Unique patients involved in near-duplicate caption pairs",
            f"{len(subjects_near)} / {n_patients} ({pct_near:.1f}%)",
        ],
        [
            "A4",
            "Test patients with near-duplicate caption linked to TRAIN",
            f"{len(train_linked_near)} / {len(test_pts)} ({pct_train_near:.1f}%)",
        ],
        [
            "A5",
            "Test patients with EXACT duplicate caption linked to TRAIN",
            f"{len(train_linked_exact)} / {len(test_pts)} ({pct_train_exact:.1f}%)",
        ],
        [
            "B1",
            "Cross-split EXACT duplicate original report text (findings+impression)",
            (
                f"{len(cross_text_groups)} groups; "
                f"{sum(g['num_patients'] for g in cross_text_groups)} patients total"
            ),
        ],
        [
            "B2",
            "Exact-text groups spanning train+val+test",
            f"{n_span_all} groups",
        ],
        [
            "B3",
            "Exact-text groups where ALL cases heuristic-normal",
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
    if img995:
        pd.DataFrame(img995).to_csv(
            os.path.join(out_dir, "06b_visually_similar_image_pairs_ge_0.995.csv"),
            index=False,
        )
    pd.DataFrame(
        [[row[1], row[2]] for row in summary[1:]], columns=["Metric", "Result"]
    ).to_csv(os.path.join(out_dir, "01_summary.csv"), index=False)

    print("\n=== MIMIC REVIEWER SUMMARY ===")
    for row in summary:
        print(row)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
