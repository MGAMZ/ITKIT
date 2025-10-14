import os

from ..base_convert import StandardFileFormatter


class LiTSFormatter(StandardFileFormatter):
    """Convert LiTS volumes and segmentations into ITKIT's standard layout."""

    VOLUME_PREFIX = "volume-"
    SEGMENTATION_PREFIX = "segmentation-"
    SUPPORTED_EXTS = (".nii.gz", ".nii")

    @staticmethod
    def _series_id(image_path: str | None, label_path: str | None) -> str:
        """Use the shared numeric suffix as the series identifier."""

        candidate = image_path or label_path
        if candidate is None:
            raise ValueError("Either image_path or label_path must be provided to derive series_id.")

        basename = os.path.basename(candidate)
        for prefix in (LiTSFormatter.VOLUME_PREFIX, LiTSFormatter.SEGMENTATION_PREFIX):
            if basename.startswith(prefix):
                basename = basename[len(prefix) :]
                break

        for ext in LiTSFormatter.SUPPORTED_EXTS:
            if basename.endswith(ext):
                return basename[: -len(ext)]

        return os.path.splitext(basename)[0]

    def tasks(self) -> list[tuple[str | None, str | None, str, str, tuple[float, float, float] | None, tuple[int, int, int] | None]]:
        spacing: tuple[float, float, float] | None = (
            tuple(self.args.spacing) if self.args.spacing is not None else None
        )
        size: tuple[int, int, int] | None = (
            tuple(self.args.size) if self.args.size is not None else None
        )

        task_list: list[tuple[str | None, str | None, str, str, tuple[float, float, float] | None, tuple[int, int, int] | None]] = []

        for root, _, files in os.walk(self.args.data_root):
            for filename in sorted(files):
                if not self._is_volume_file(filename):
                    continue

                image_path = os.path.join(root, filename)
                label_path = self._find_label_path(root, filename)
                series_id = self._series_id(image_path, label_path)

                task_list.append(
                    (
                        image_path,
                        label_path,
                        self.args.dest_root,
                        series_id,
                        spacing,
                        size,
                    )
                )

        return task_list

    @classmethod
    def _is_volume_file(cls, filename: str) -> bool:
        return filename.startswith(cls.VOLUME_PREFIX) and any(
            filename.endswith(ext) for ext in cls.SUPPORTED_EXTS
        )

    @classmethod
    def _strip_extension(cls, filename: str) -> tuple[str, str]:
        for ext in cls.SUPPORTED_EXTS:
            if filename.endswith(ext):
                return filename[: -len(ext)], ext
        stem, ext = os.path.splitext(filename)
        return stem, ext

    def _find_label_path(self, root: str, volume_filename: str) -> str | None:
        volume_suffix, volume_ext = self._strip_extension(volume_filename)

        if not volume_suffix.startswith(self.VOLUME_PREFIX):
            # Fallback: already stripped prefix earlier when identifying volumes.
            suffix = volume_suffix
        else:
            suffix = volume_suffix[len(self.VOLUME_PREFIX) :]

        candidate_names = []

        if volume_ext:
            candidate_names.append(f"{self.SEGMENTATION_PREFIX}{suffix}{volume_ext}")

        for label_ext in self.SUPPORTED_EXTS:
            candidate = f"{self.SEGMENTATION_PREFIX}{suffix}{label_ext}"
            if candidate not in candidate_names:
                candidate_names.append(candidate)

        for candidate in candidate_names:
            candidate_path = os.path.join(root, candidate)
            if os.path.exists(candidate_path):
                return candidate_path

        return None


if __name__ == "__main__":
    formatter = LiTSFormatter()
    formatter.execute()
