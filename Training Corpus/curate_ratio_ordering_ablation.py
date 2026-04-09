#!/usr/bin/env python3
"""
Curate deterministic ratio x ordering datasets for the ordering ablation.

The source memberships are fixed by the existing checked-in curriculum files.
This script only reorders those memberships into a new output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


BENEFICIAL_SOURCE = "hammh0a/Hala-4.6M-SFT"
PROJECT_SHUFFLE_SEED = 20260409
ORDERINGS = ("beneficial-first", "refusal-first", "random", "interleaved")
SOURCE_DATASETS = {
    "80ben_20ref": "hala_8000_ref_2000.jsonl",
    "90ben_10ref": "hala_9000_ref_1000.jsonl",
    "95ben_5ref": "hala_9500_ref_500.jsonl",
}
EXPECTED_COUNTS = {
    "80ben_20ref": {"beneficial": 8000, "refusal": 2000},
    "90ben_10ref": {"beneficial": 9000, "refusal": 1000},
    "95ben_5ref": {"beneficial": 9500, "refusal": 500},
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Curate deterministic ratio x ordering datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "curriculumLearningTrainingCorpus",
        help="Directory containing the canonical ratio datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "ratio_ordering_ablation",
        help="Directory where curated datasets will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing curated datasets.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def classify_row(row: dict) -> str:
    return "beneficial" if row.get("source") == BENEFICIAL_SOURCE else "refusal"


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    beneficial = [row for row in rows if classify_row(row) == "beneficial"]
    refusal = [row for row in rows if classify_row(row) == "refusal"]
    return beneficial, refusal


def derived_seed(*parts: str) -> int:
    payload = "|".join((str(PROJECT_SHUFFLE_SEED), *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def shuffled_copy(rows: list[dict], *seed_parts: str) -> list[dict]:
    result = list(rows)
    random.Random(derived_seed(*seed_parts)).shuffle(result)
    return result


def interleave_rows(beneficial: list[dict], refusal: list[dict]) -> list[dict]:
    total_beneficial = len(beneficial)
    total_refusal = len(refusal)
    total = total_beneficial + total_refusal

    if total == 0:
        return []

    output: list[dict] = []
    beneficial_idx = 0
    refusal_idx = 0

    for position in range(total):
        if beneficial_idx == total_beneficial:
            output.append(refusal[refusal_idx])
            refusal_idx += 1
            continue
        if refusal_idx == total_refusal:
            output.append(beneficial[beneficial_idx])
            beneficial_idx += 1
            continue

        beneficial_target = ((position + 1) * total_beneficial) / total
        refusal_target = ((position + 1) * total_refusal) / total
        beneficial_deficit = beneficial_target - beneficial_idx
        refusal_deficit = refusal_target - refusal_idx

        if refusal_deficit > beneficial_deficit:
            output.append(refusal[refusal_idx])
            refusal_idx += 1
        else:
            output.append(beneficial[beneficial_idx])
            beneficial_idx += 1

    return output


def build_ordered_rows(
    ratio_label: str,
    ordering: str,
    beneficial: list[dict],
    refusal: list[dict],
) -> list[dict]:
    if ordering == "beneficial-first":
        return list(beneficial) + list(refusal)
    if ordering == "refusal-first":
        return list(refusal) + list(beneficial)
    if ordering == "random":
        return shuffled_copy(list(beneficial) + list(refusal), ratio_label, ordering)
    if ordering == "interleaved":
        return interleave_rows(list(beneficial), list(refusal))
    raise ValueError(f"Unsupported ordering: {ordering}")


def count_classes(rows: list[dict]) -> dict[str, int]:
    counts = {"beneficial": 0, "refusal": 0}
    for row in rows:
        counts[classify_row(row)] += 1
    return counts


def count_runs(rows: list[dict]) -> int:
    if not rows:
        return 0

    runs = 1
    previous = classify_row(rows[0])
    for row in rows[1:]:
        current = classify_row(row)
        if current != previous:
            runs += 1
            previous = current
    return runs


def first_index(rows: list[dict], target_class: str) -> int | None:
    for index, row in enumerate(rows):
        if classify_row(row) == target_class:
            return index
    return None


def last_index(rows: list[dict], target_class: str) -> int | None:
    for index in range(len(rows) - 1, -1, -1):
        if classify_row(rows[index]) == target_class:
            return index
    return None


def ensure_valid_ordering(ordering: str, rows: list[dict], expected_counts: dict[str, int]) -> None:
    counts = count_classes(rows)
    if counts != expected_counts:
        raise ValueError(f"{ordering}: expected counts {expected_counts}, got {counts}")

    beneficial_first = first_index(rows, "beneficial")
    refusal_first = first_index(rows, "refusal")
    beneficial_last = last_index(rows, "beneficial")
    refusal_last = last_index(rows, "refusal")
    runs = count_runs(rows)

    if ordering == "beneficial-first":
        if beneficial_first != 0 or refusal_first != expected_counts["beneficial"]:
            raise ValueError(f"{ordering}: block layout is incorrect")
        if runs != 2:
            raise ValueError(f"{ordering}: expected exactly 2 class runs, got {runs}")
        return

    if ordering == "refusal-first":
        if refusal_first != 0 or beneficial_first != expected_counts["refusal"]:
            raise ValueError(f"{ordering}: block layout is incorrect")
        if runs != 2:
            raise ValueError(f"{ordering}: expected exactly 2 class runs, got {runs}")
        return

    if ordering == "random":
        if runs <= 2:
            raise ValueError(f"{ordering}: ordering still looks block-ordered")
        if beneficial_first is None or refusal_first is None:
            raise ValueError(f"{ordering}: missing one class entirely")
        return

    if ordering == "interleaved":
        if runs <= 2:
            raise ValueError(f"{ordering}: ordering still looks block-ordered")
        half = len(rows) // 2
        first_half = rows[:half]
        second_half = rows[half:]
        first_half_counts = count_classes(first_half)
        second_half_counts = count_classes(second_half)
        if 0 in first_half_counts.values() or 0 in second_half_counts.values():
            raise ValueError(f"{ordering}: both classes must appear in each half")
        if refusal_last != len(rows) - 1 and beneficial_last != len(rows) - 1:
            raise ValueError(f"{ordering}: invalid trailing index state")
        return

    raise ValueError(f"Unsupported ordering: {ordering}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    ratio_label: str,
    ordering: str,
    source_file: Path,
    dataset_file: Path,
    rows: list[dict],
) -> dict:
    counts = count_classes(rows)
    return {
        "ratio": ratio_label,
        "ordering": ordering,
        "source_file": str(source_file.relative_to(source_file.parents[1])),
        "dataset_file": str(dataset_file.relative_to(dataset_file.parents[2])),
        "project_shuffle_seed": PROJECT_SHUFFLE_SEED,
        "total_rows": len(rows),
        "beneficial_count": counts["beneficial"],
        "refusal_count": counts["refusal"],
        "class_runs": count_runs(rows),
        "first_beneficial_index": first_index(rows, "beneficial"),
        "first_refusal_index": first_index(rows, "refusal"),
        "last_beneficial_index": last_index(rows, "beneficial"),
        "last_refusal_index": last_index(rows, "refusal"),
        "dataset_sha256": sha256_file(dataset_file),
    }


def curate_ratio(
    ratio_label: str,
    source_path: Path,
    output_dir: Path,
    force: bool,
) -> list[dict]:
    rows = load_jsonl(source_path)
    beneficial, refusal = split_rows(rows)
    expected_counts = EXPECTED_COUNTS[ratio_label]

    if len(rows) != 10000:
        raise ValueError(f"{source_path} must contain 10000 rows, found {len(rows)}")
    if {"beneficial": len(beneficial), "refusal": len(refusal)} != expected_counts:
        raise ValueError(
            f"{source_path} does not match expected counts {expected_counts}"
        )

    ratio_output_dir = output_dir / ratio_label
    ratio_output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict] = []
    for ordering in ORDERINGS:
        dataset_path = ratio_output_dir / f"{ordering}.jsonl"
        manifest_path = ratio_output_dir / f"{ordering}.manifest.json"

        if not force and (dataset_path.exists() or manifest_path.exists()):
            raise FileExistsError(
                f"{dataset_path} or {manifest_path} already exists. Use --force to overwrite."
            )

        ordered_rows = build_ordered_rows(ratio_label, ordering, beneficial, refusal)
        ensure_valid_ordering(ordering, ordered_rows, expected_counts)
        write_jsonl(dataset_path, ordered_rows)

        manifest = build_manifest(ratio_label, ordering, source_path, dataset_path, ordered_rows)
        write_json(manifest_path, manifest)
        manifests.append(manifest)

    return manifests


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_payload = {
        "project_shuffle_seed": PROJECT_SHUFFLE_SEED,
        "source_datasets": SOURCE_DATASETS,
        "datasets": [],
    }

    for ratio_label, filename in SOURCE_DATASETS.items():
        source_path = args.input_dir / filename
        manifests = curate_ratio(
            ratio_label=ratio_label,
            source_path=source_path,
            output_dir=args.output_dir,
            force=args.force,
        )
        index_payload["datasets"].extend(manifests)

    write_json(args.output_dir / "index.json", index_payload)

    print("Curated datasets written to:", args.output_dir)
    print("Datasets created:", len(index_payload["datasets"]))


if __name__ == "__main__":
    main()
