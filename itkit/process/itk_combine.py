import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from itkit.process.base_processor import BaseITKProcessor
from itkit.process.metadata_models import SeriesMetadata


@dataclass(frozen=True)
class SourceSpec:
    name: str
    folder: Path


@dataclass(frozen=True)
class MappingRule:
    source_name: str
    source_labels: tuple[int, ...]
    target_label: int


def _parse_sources(source_args: list[str]) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    seen_names: set[str] = set()
    for item in source_args:
        if "=" not in item:
            raise ValueError(f"Invalid source format: {item}. Expected name=/path/to/labels")
        name, folder = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid source name in: {item}")
        if name in seen_names:
            raise ValueError(f"Duplicate source name: {name}")
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Source folder not found: {folder_path}")
        sources.append(SourceSpec(name=name, folder=folder_path))
        seen_names.add(name)
    return sources


def _parse_mapping_rule(rule: str) -> MappingRule:
    if "->" not in rule or ":" not in rule:
        raise ValueError(f"Invalid mapping rule: {rule}. Expected <source>:<src_labels>-><target>")
    left, target_str = rule.split("->", 1)
    source_name, labels_str = left.split(":", 1)
    source_name = source_name.strip()
    labels_str = labels_str.strip()
    target_str = target_str.strip()
    if not source_name or not labels_str or not target_str:
        raise ValueError(f"Invalid mapping rule: {rule}. Expected <source>:<src_labels>-><target>")

    try:
        target_label = int(target_str)
    except ValueError as exc:
        raise ValueError(f"Invalid target label in rule: {rule}") from exc

    label_parts = [p.strip() for p in labels_str.split(",") if p.strip()]
    if not label_parts:
        raise ValueError(f"No source labels specified in rule: {rule}")

    source_labels: list[int] = []
    for part in label_parts:
        try:
            source_labels.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid source label '{part}' in rule: {rule}") from exc

    return MappingRule(source_name=source_name, source_labels=tuple(source_labels), target_label=target_label)


class CombineProcessor(BaseITKProcessor):
    def __init__(
        self,
        sources: list[SourceSpec],
        dest_folder: Path,
        mapping_rules: list[MappingRule],
        mp: bool = False,
        workers: int | None = None,
    ):
        super().__init__(task_description="Combining labels", mp=mp, workers=workers)
        self.sources = sources
        self.dest_folder = dest_folder
        self.mapping_rules = mapping_rules
        self.source_index = {src.name: idx for idx, src in enumerate(self.sources)}

    def get_items_to_process(self) -> list[tuple[str, list[str]]]:
        source_files: dict[str, dict[str, str]] = {}
        for src in self.sources:
            files = {p.name: str(p) for p in src.folder.glob("*.mha")}
            source_files[src.name] = files

        common_names = None
        for files in source_files.values():
            names = set(files.keys())
            common_names = names if common_names is None else common_names & names
        if not common_names:
            return []

        items = []
        for name in sorted(common_names):
            paths = [source_files[src.name][name] for src in self.sources]
            items.append((name, paths))
        return items

    def process_one(self, args: tuple[str, list[str]]) -> SeriesMetadata | None:
        name, paths = args
        images = [sitk.ReadImage(p) for p in paths]
        base_size = images[0].GetSize()
        base_spacing = images[0].GetSpacing()

        for idx, image in enumerate(images[1:], start=1):
            if image.GetSize() != base_size:
                raise ValueError(f"Size mismatch for {name}: {paths[0]} vs {paths[idx]}")
            if not np.allclose(image.GetSpacing(), base_spacing):
                raise ValueError(f"Spacing mismatch for {name}: {paths[0]} vs {paths[idx]}")

        arrays = [sitk.GetArrayFromImage(img) for img in images]
        output = np.zeros(arrays[0].shape, dtype=np.uint8)

        for rule in self.mapping_rules:
            src_idx = self.source_index[rule.source_name]
            src_arr = arrays[src_idx]
            mask = np.isin(src_arr, rule.source_labels)
            mask = mask & (output == 0)
            output[mask] = rule.target_label

        out_image = sitk.GetImageFromArray(output)
        out_image.CopyInformation(images[0])

        output_path = self.dest_folder / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out_image, str(output_path), useCompression=True)

        return SeriesMetadata.from_sitk_image(out_image, name)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="itk_combine",
        description=(
            "Combine multiple label folders by intersecting filenames and merging labels "
            "according to ordered mapping rules."
        ),
    )
    parser.add_argument(
        "-i", "--source",
        action="append",
        required=True,
        help="Label source in form name=/path/to/label_folder (repeatable)",
    )
    parser.add_argument(
        "--map",
        dest="mapping_rules",
        action="append",
        required=True,
        help="Mapping rule in form `<source>:<src_labels>-><target>`, e.g., `A:1,2->3` (repeatable)",
    )
    parser.add_argument(
        "-o", "dest_folder",
        type=Path,
        help="Destination folder for combined labels",
    )
    parser.add_argument("--mp", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    return parser.parse_args()


def main():
    args = parse_args()

    sources = _parse_sources(args.source)
    rules = [_parse_mapping_rule(rule) for rule in args.mapping_rules]

    source_names = {s.name for s in sources}
    for rule in rules:
        if rule.source_name not in source_names:
            raise ValueError(f"Mapping rule references unknown source: {rule.source_name}")

    if not rules:
        raise ValueError("At least one mapping rule is required.")

    dest_folder = args.dest_folder.expanduser().resolve()
    os.makedirs(dest_folder, exist_ok=True)

    print("Combining label sources:")
    for src in sources:
        print(f"  - {src.name}: {src.folder}")
    print("Mapping rules (ordered, earlier has higher priority):")
    for rule in rules:
        print(f"  - {rule.source_name}: {list(rule.source_labels)} -> {rule.target_label}")
    print(f"Output: {dest_folder}")
    print(f"Multiprocessing: {args.mp} | Workers: {args.workers}")

    processor = CombineProcessor(
        sources=sources,
        dest_folder=dest_folder,
        mapping_rules=rules,
        mp=args.mp,
        workers=args.workers,
    )

    processor.process("Combining labels")
    processor.save_meta(dest_folder / "meta.json")

    print("Combine completed.")


if __name__ == "__main__":
    main()
