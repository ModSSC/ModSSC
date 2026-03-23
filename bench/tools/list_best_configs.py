from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

EXCLUDED_METHODS: dict[str, str] = {
    "tsvm": "Unsupported in the standard sweep.",
    "lazy_random_walk": "Excluded due to memory pressure and algorithmic limitations.",
}

ADULT_TRANSDUCTIVE_TO_PATCH_METHODS: dict[str, str] = {
    "appnp": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "chebnet": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "gat": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "gcn": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "gcnii": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "grafn": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "grand": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "graph_mincuts": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "graphsage": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "h_gcn": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "n_gcn": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "planetoid": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
    "sgc": "Adult transductive full-graph GNN is GPU-memory fragile on Jean Zay.",
}

NONE_LIKE = {"", "none", "null", "false"}


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _path_meta(path: Path, root: Path) -> dict[str, str] | None:
    try:
        percentage, kind, method, modality, filename = path.relative_to(root).parts
    except ValueError:
        return None
    except Exception:
        return None
    if not filename.endswith(".yaml"):
        return None
    return {
        "percentage": percentage,
        "kind": kind,
        "method": method,
        "modality": modality,
        "dataset": filename[:-5],
    }


def _preprocess_step_ids(data: dict[str, Any]) -> set[str]:
    steps = (((data.get("preprocess") or {}).get("plan") or {}).get("steps")) or []
    ids: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or step.get("step_id") or "").strip()
        if step_id:
            ids.add(step_id)
    return ids


def _classifier_meta(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model = (data.get("method") or {}).get("model") or {}
    classifier_id = str(model.get("classifier_id") or "").strip()
    classifier_params = model.get("classifier_params") or {}
    if not isinstance(classifier_params, dict):
        classifier_params = {}
    return classifier_id, classifier_params


def classify_config(path: Path, *, root: Path) -> tuple[str, str]:
    meta = _path_meta(path, root)
    if meta is None:
        return "OK", "Non-standard path; kept by default."

    method = meta["method"]
    if method in EXCLUDED_METHODS:
        return "EXCLUDED", EXCLUDED_METHODS[method]

    if (
        meta["kind"] == "transductive"
        and meta["modality"] == "tabular"
        and meta["dataset"] == "adult"
        and method in ADULT_TRANSDUCTIVE_TO_PATCH_METHODS
    ):
        return "TO_PATCH", ADULT_TRANSDUCTIVE_TO_PATCH_METHODS[method]

    data = _load_config(path)
    step_ids = _preprocess_step_ids(data)
    classifier_id, classifier_params = _classifier_meta(data)

    if "text.sentence_transformer" in step_ids:
        return "OFFLINE_FIXABLE", "Requires sentence-transformers/all-MiniLM-L6-v2 in cache."
    if "audio.wav2vec2" in step_ids:
        return "OFFLINE_FIXABLE", "Requires torchaudio WAV2VEC2_BASE checkpoint in cache."
    if "vision.openclip" in step_ids:
        return "OFFLINE_FIXABLE", "Requires OpenCLIP checkpoint in cache."
    if classifier_id == "audio_pretrained":
        return "OFFLINE_FIXABLE", "Requires torchaudio WAV2VEC2_BASE checkpoint in cache."
    if classifier_id == "image_pretrained":
        weights = str(classifier_params.get("weights") or "").strip().lower()
        if weights and weights not in NONE_LIKE:
            return "OFFLINE_FIXABLE", "Requires torchvision pretrained weights in cache."

    return "OK", "Eligible for the standard sweep."


def _matches(value: str, selected: set[str] | None) -> bool:
    if not selected:
        return True
    return value in selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List best benchmark configs for the standard sweep."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("bench/configs/best"),
        help="Root directory containing best benchmark configs.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Filter by dataset id.")
    parser.add_argument("--modality", action="append", default=[], help="Filter by modality.")
    parser.add_argument("--kind", action="append", default=[], help="Filter by benchmark kind.")
    parser.add_argument("--method", action="append", default=[], help="Filter by method id.")
    parser.add_argument("--percentage", action="append", default=[], help="Filter by percentage.")
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Explicit statuses to keep. Defaults to OK and OFFLINE_FIXABLE.",
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Keep every status, including TO_PATCH and EXCLUDED.",
    )
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Prefix each emitted path with its classification status.",
    )
    parser.add_argument(
        "--show-reason",
        action="store_true",
        help="Include the classification reason before each emitted path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    dataset_filter = set(args.dataset)
    modality_filter = set(args.modality)
    kind_filter = set(args.kind)
    method_filter = set(args.method)
    percentage_filter = set(args.percentage)

    if args.all_statuses:
        status_filter: set[str] | None = None
    elif args.status:
        status_filter = set(args.status)
    else:
        status_filter = {"OK", "OFFLINE_FIXABLE"}

    for path in sorted(root.rglob("*.yaml")):
        meta = _path_meta(path, root)
        if meta is None:
            continue
        if not _matches(meta["dataset"], dataset_filter):
            continue
        if not _matches(meta["modality"], modality_filter):
            continue
        if not _matches(meta["kind"], kind_filter):
            continue
        if not _matches(meta["method"], method_filter):
            continue
        if not _matches(meta["percentage"], percentage_filter):
            continue

        status, reason = classify_config(path, root=root)
        if status_filter is not None and status not in status_filter:
            continue

        rel_path = path.as_posix()
        if args.show_status and args.show_reason:
            print(f"{status}\t{reason}\t{rel_path}")
        elif args.show_status:
            print(f"{status}\t{rel_path}")
        elif args.show_reason:
            print(f"{reason}\t{rel_path}")
        else:
            print(rel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
