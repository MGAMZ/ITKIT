import argparse
import shutil
from pathlib import Path


def reorganize_msd(root_dir):
    """
    Reorganizes MSD dataset structure from TaskXX/imagesTr, TaskXX/imagesTs, TaskXX/labelsTr
    to TaskXX/image and TaskXX/label.
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists():
        print(f"Error: Path {root_dir} does not exist.")
        return

    # Iterate through each folder in the root directory
    task_folders = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Task")]

    if not task_folders:
        print(f"No Task folders found in {root_path}")
        return

    for task_dir in task_folders:
        print(f"\nProcessing {task_dir.name}...")

        image_dir = task_dir / "image"
        label_dir = task_dir / "label"

        # Create image directory
        image_dir.mkdir(parents=True, exist_ok=True)

        # 1. Move imagesTr and imagesTs to 'image'
        for sub_name in ["imagesTr", "imagesTs"]:
            src_dir = task_dir / sub_name
            if src_dir.exists() and src_dir.is_dir():
                print(f"  Moving {sub_name} content to image/")
                for item in src_dir.iterdir():
                    if item.is_file():
                        target = image_dir / item.name
                        if target.exists():
                            print(f"    Warning: {item.name} already exists in 'image', skipping.")
                        else:
                            shutil.move(str(item), str(target))

                # Try to remove original dir if empty
                try:
                    src_dir.rmdir()
                except OSError:
                    print(f"    Note: {sub_name} is not empty, keeping directory.")

        # 2. Rename labelsTr to 'label'
        labels_tr = task_dir / "labelsTr"
        if labels_tr.exists() and labels_tr.is_dir():
            if label_dir.exists():
                print("  Moving labelsTr content to existing label/")
                for item in labels_tr.iterdir():
                    if item.is_file():
                        target = label_dir / item.name
                        if target.exists():
                            print(f"    Warning: {item.name} already exists in 'label', skipping.")
                        else:
                            shutil.move(str(item), str(target))
                try:
                    labels_tr.rmdir()
                except OSError:
                    print("    Note: labelsTr is not empty, keeping directory.")
            else:
                print("  Renaming labelsTr to label")
                labels_tr.rename(label_dir)

    print("\nReorganization completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize MSD dataset structure.")
    parser.add_argument("root_dir", type=str, help="Root directory containing MSD Task folders (e.g. Task01_BrainTumour, etc.)")

    args = parser.parse_args()
    reorganize_msd(args.root_dir)
