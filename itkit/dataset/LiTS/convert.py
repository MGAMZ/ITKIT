import os
from glob import glob

from itkit.dataset.base_convert import StandardFileFormatter


class LiTSFormatter(StandardFileFormatter):
    """Convert LiTS volumes and segmentations into ITKIT's standard layout.
    
    Expects all volume-*.nii[.gz] and segmentation-*.nii[.gz] files 
    in the same directory (flat structure).
    """

    @staticmethod
    def _series_id(image_path: str | None, label_path: str | None) -> str:
        """Extract numeric ID from filename: volume-123.nii -> 123"""
        path = image_path or label_path
        assert path is not None, "Either image_path or label_path must be provided."
        basename = os.path.basename(path)
        # Remove 'volume-' or 'segmentation-' prefix and file extension
        name = basename.replace("volume-", "").replace("segmentation-", "")
        name = name.replace(".nii.gz", "").replace(".nii", "")
        return name

    def tasks(self):
        spacing = tuple(self.args.spacing) if self.args.spacing else None
        size = tuple(self.args.size) if self.args.size else None
        task_list = []

        # Find all volume files (supports both .nii and .nii.gz)
        volume_pattern = os.path.join(self.args.data_root, "volume-*.nii*")
        volume_files = sorted(glob(volume_pattern))

        for volume_path in volume_files:
            # Construct corresponding segmentation filename
            basename = os.path.basename(volume_path)
            seg_filename = basename.replace("volume-", "segmentation-")
            seg_path = os.path.join(self.args.data_root, seg_filename)
            
            # Use segmentation path only if it exists
            label_path = seg_path if os.path.exists(seg_path) else None
            series_id = self._series_id(volume_path, label_path)

            task_list.append((
                volume_path,
                label_path,
                self.args.dest_root,
                series_id,
                spacing,
                size,
            ))

        return task_list


if __name__ == "__main__":
    formatter = LiTSFormatter()
    formatter.execute()
