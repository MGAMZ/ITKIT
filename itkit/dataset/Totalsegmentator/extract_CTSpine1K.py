import os, argparse

from itkit.process.itk_extract import ExtractProcessor


TSD_TO_CTSPINE1K_LABEL_MAPPING = {
    # Cervical 颈椎
    93:1,
    94:2,
    95:3,
    96:4,
    97:5,
    98:6,
    99:7,
    # Thoracic 胸椎
    106:8,
    110:9,
    111:10,
    112:11,
    113:12,
    114:13,
    115:14,
    116:15,
    117:16,
    107:17,
    108:18,
    109:19,
    # Lumbar 腰椎
    100:20,
    101:21,
    102:22,
    103:23,
    104:24,
    # Sacral 骶椎
    105:25,
}


def parse_args():
    parser = argparse.ArgumentParser(description='ITK label extraction for TotalSegmentator CTSpine1K')
    parser.add_argument('tsd_label', type=str, help='Folder containing input label mhas')
    parser.add_argument('output_folder', type=str, help='Folder containing input label mhas')
    parser.add_argument('--mp', action='store_true', default=False, help='Use multiprocessing')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    processor = ExtractProcessor(
        source_folder = args.tsd_label,
        dest_folder = args.output_folder,
        label_mapping = TSD_TO_CTSPINE1K_LABEL_MAPPING,
        mp = args.mp,
    )
    processor.process()
    processor.save_meta(os.path.join(args.output_folder, "extract_meta.json"))
