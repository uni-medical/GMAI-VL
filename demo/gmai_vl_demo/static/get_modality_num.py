import argparse
import json
import os
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


DEFAULT_DATASET_ROOT = "data/SA-Med2D-20M/raw/SAMed2Dv1"
MAIN_MODALITY_DISPLAY_NAMES = {
    "X": "X-Ray",
}
MR_MAIN_MODALITY_RULES = (
    ("MR-FLAIR", ("mr_flair",)),
    ("MR-T1", ("mr_t1", "mr_t1c", "mr_t1ce", "mr_t1gd", "mr_t1w", "mr_mprage")),
    ("MR-T2", ("mr_t2", "mr_t2w", "mr_pd")),
    ("MR-DWI/ADC", ("mr_dwi", "mr_adc")),
    ("MR-PWI", ("mr_cbf", "mr_cbv", "mr_mtt", "mr_rcbf", "mr_rcbv", "mr_tmax", "mr_ttp")),
    ("MR-CMR", ("mr_cmr", "mr_lge")),
)
MAIN_ORGAN_RULES = (
    ("adrenal_gland", ("adrenal",)),
    ("airway", ("airway",)),
    ("aorta", ("aorta",)),
    ("artery", ("artery", "carotid_vessel_wall", "pulmonary_artery")),
    ("vein", ("vena", "vena_cava", "iliac_vena", "portal_vein")),
    ("bladder", ("bladder", "urinary_bladder")),
    ("bone", ("bone", "clavicula", "femur", "hip", "humerus", "sacrum", "scapula", "vertebrae")),
    ("brain", ("brain", "brainstem", "matter_tracts", "temporal_lobes")),
    ("colon", ("colon", "rectum")),
    ("duodenum", ("duodenum",)),
    ("esophagus", ("esophagus",)),
    ("eye", ("eye", "lens")),
    ("face", ("face",)),
    ("gallbladder", ("gallbladder",)),
    ("heart", ("heart", "myocardial", "myocardium", "no_reflow")),
    ("intestine", ("intestine", "small_bowel")),
    ("kidney", ("kidney",)),
    ("liver", ("liver", "hepatic")),
    ("lung", ("lung", "pulmonary_embolism")),
    ("mandible", ("mandible",)),
    ("middle_ear", ("middle_ear", "inner_ear")),
    ("muscle", ("autochthon", "gluteus", "iliopsoas")),
    ("pancreas", ("pancreas", "pancreatic")),
    ("parotid_gland", ("parotid",)),
    ("prostate", ("prostate",)),
    ("rib", ("rib_",)),
    ("spinal_cord", ("spinal_cord", "cord")),
    ("spleen", ("spleen",)),
    ("stomach", ("stomach",)),
    ("trachea", ("trachea",)),
    ("uterus", ("uterus",)),
)


def parse_samed2d_name(path: str) -> Optional[Dict[str, str]]:
    name = path.rsplit("/", 1)[-1]
    if not name.lower().endswith(".png"):
        return None

    parts = name[:-4].split("--", 4)
    if len(parts) < 4:
        return None

    modality = parts[0]
    return {
        "modality": modality,
        "main_modality": get_main_modality(modality),
        "dataset": parts[1],
    }


def get_main_modality(modality: str) -> str:
    modality_lower = modality.lower()
    if modality_lower.startswith("mr"):
        for main_modality, sub_modalities in MR_MAIN_MODALITY_RULES:
            if modality_lower in sub_modalities:
                return main_modality
        return "MR-Other"

    prefix = modality.split("_", 1)[0].upper()
    return MAIN_MODALITY_DISPLAY_NAMES.get(prefix, prefix)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_image_paths_from_metadata(obj) -> Iterable[str]:
    if isinstance(obj, str):
        if obj.lower().endswith(".png") and "--" in obj:
            yield obj
        return

    if isinstance(obj, list):
        for item in obj:
            yield from iter_image_paths_from_metadata(item)
        return

    if isinstance(obj, dict):
        for value in obj.values():
            yield from iter_image_paths_from_metadata(value)


def collect_image_paths(dataset_path: Path, metadata_filename: str) -> List[str]:
    images_dir = dataset_path / "images"
    if images_dir.is_dir():
        return [
            entry.name
            for entry in os.scandir(images_dir)
            if entry.is_file() and entry.name.lower().endswith(".png")
        ]

    metadata_path = dataset_path / metadata_filename
    metadata = load_json(metadata_path)
    return list(iter_image_paths_from_metadata(metadata))


def is_non_union_label(label_name: str) -> bool:
    return bool(label_name) and not label_name.startswith("union_")


def normalize_main_organ(label_name: str) -> Optional[str]:
    if label_name.startswith("class") or label_name in {"left", "right"}:
        return None

    for organ_name, patterns in MAIN_ORGAN_RULES:
        if any(pattern in label_name for pattern in patterns):
            return organ_name

    if any(
        pattern in label_name
        for pattern in (
            "COVID_lesion",
            "cancer",
            "edema",
            "hemorrhage",
            "lesion",
            "necrosis",
            "sclerosis",
            "schwannoma",
            "tumor",
        )
    ):
        return "lesion"

    return None


def collect_mapping_statistics(mapping_json_path: Path, image_datasets: Sequence[str]) -> Dict[str, object]:
    raw_mapping = load_json(mapping_json_path)
    if not isinstance(raw_mapping, dict):
        raise ValueError("The class mapping JSON top level should be a dict.")

    mapping_dataset_count = 0
    raw_entry_count = 0
    non_union_label_entry_count = 0
    non_union_labels: List[str] = []
    main_organ_to_labels: Dict[str, Set[str]] = {}

    for dataset_name in sorted(image_datasets):
        class_mapping = raw_mapping.get(dataset_name)
        if not isinstance(class_mapping, dict):
            continue
        mapping_dataset_count += 1
        for label_name in class_mapping:
            raw_entry_count += 1
            if is_non_union_label(label_name):
                non_union_label_entry_count += 1
                non_union_labels.append(label_name)
                main_organ = normalize_main_organ(label_name)
                if main_organ:
                    main_organ_to_labels.setdefault(main_organ, set()).add(label_name)

    return {
        "mapping_dataset_count": mapping_dataset_count,
        "raw_entry_count": raw_entry_count,
        "non_union_label_entry_count": non_union_label_entry_count,
        "organ_count": len(main_organ_to_labels),
        "main_organ_counts": {
            main_organ: len(labels)
            for main_organ, labels in sorted(main_organ_to_labels.items())
        },
        "non_union_labels": non_union_labels,
    }


def collect_image_statistics(dataset_path: Path, metadata_filename: str) -> Dict[str, object]:
    modality_tree: Dict[str, Counter] = {}
    image_datasets: Set[str] = set()
    total_images = 0

    for image_path in collect_image_paths(dataset_path, metadata_filename):
        parsed = parse_samed2d_name(image_path)
        if not parsed:
            continue
        total_images += 1
        image_datasets.add(parsed["dataset"])
        modality_tree.setdefault(parsed["main_modality"], Counter())[parsed["modality"]] += 1

    return {
        "total_images": total_images,
        "modality_tree": modality_tree,
        "image_datasets": sorted(image_datasets),
    }


def print_modality_table(modality_tree: Dict[str, Counter]) -> None:
    print(f"{'Main modality':<12} | {'Images':>12} | Sub-modalities")
    print("-" * 95)

    for main_modality in sorted(modality_tree):
        sub_counts = modality_tree[main_modality]
        total = sum(sub_counts.values())
        sub_items = ", ".join(
            f"{sub_modality}({count:,})"
            for sub_modality, count in sorted(sub_counts.items())
        )
        wrapped_lines = textwrap.wrap(sub_items, width=62) or [""]

        print(f"{main_modality:<12} | {total:>12,} | {wrapped_lines[0]}")
        for line in wrapped_lines[1:]:
            print(f"{'':<12} | {'':>12} | {line}")


def print_list(items: Sequence[str], cols: int = 6) -> None:
    if not items:
        print("None")
        return

    col_width = max(22, min(42, max(len(item) for item in items) + 2))
    for i in range(0, len(items), cols):
        row = items[i : i + cols]
        print("".join(f"{item:<{col_width}}" for item in row).rstrip())


def print_main_organ_counts(main_organ_counts: Dict[str, int], cols: int = 4) -> None:
    items = [
        f"{main_organ}({count:,})"
        for main_organ, count in sorted(main_organ_counts.items())
    ]
    print_list(items, cols=cols)


def analyze_modalities_and_labels(
    dataset_path: Path,
    metadata_filename: str,
    mapping_filename: str,
) -> None:
    mapping_json_path = dataset_path / mapping_filename

    image_stats = collect_image_statistics(dataset_path, metadata_filename)
    mapping_stats = collect_mapping_statistics(mapping_json_path, image_stats["image_datasets"])
    label_preview = mapping_stats["non_union_labels"]

    print("=" * 120)
    print("SA-Med2D modality and label statistics")
    print(f"Dataset root: {dataset_path}")
    print(
        f"Images: {image_stats['total_images']:,} | "
        f"Image-source datasets: {len(image_stats['image_datasets']):,} | "
        f"Main modalities: {len(image_stats['modality_tree']):,}"
    )
    print(
        f"Matched class-mapping datasets: {mapping_stats['mapping_dataset_count']:,} | "
        f"Mapping entries: {mapping_stats['raw_entry_count']:,} | "
        f"Non-union label entries: {mapping_stats['non_union_label_entry_count']:,} | "
        f"Main organs: {mapping_stats['organ_count']:,}"
    )
    print("=" * 120)

    print("\n[Modality statistics]")
    if image_stats["modality_tree"]:
        print_modality_table(image_stats["modality_tree"])
    else:
        print("No parseable PNG image paths found.")

    print("\n[Main organs; number in parentheses is matched base labels]")
    print_main_organ_counts(mapping_stats["main_organ_counts"])

    print(f"\n[Base labels; excluding union_*; {mapping_stats['non_union_label_entry_count']:,} entries]")
    print_list(label_preview)
    print("=" * 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SA-Med2D modalities, main organs, and label coverage."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help=f"Path to SAMed2Dv1 directory. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument(
        "--metadata-filename",
        default="SAMed2D_v1.json",
        help="Metadata JSON filename used when images/ is unavailable.",
    )
    parser.add_argument(
        "--mapping-filename",
        default="SAMed2D_v1_class_mapping_id.json",
        help="Class mapping JSON filename.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_modalities_and_labels(
        Path(args.dataset_root),
        metadata_filename=args.metadata_filename,
        mapping_filename=args.mapping_filename,
    )
