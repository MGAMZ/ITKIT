"""
We recommend following official guidelines and tools from TCIA for downloading data.
https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-DownloadingtheNBIADataRetriever
"""

import argparse
import os

import pandas as pd
from tcia_utils import nbia


def update_meta_filepath(data_root: str, meta_path: str):
    df = pd.read_csv(meta_path)

    for i in range(len(df)):
        series_uid = df.loc[i, "Series UID"]
        filepath = df.loc[i, "File Location"]
        if not os.path.exists(os.path.join(data_root, str(filepath))):
            print(f"Series {series_uid} not found according to meta file path.")
            if os.path.exists(os.path.join(data_root, str(series_uid))):
                print(f"auto found {series_uid} in {data_root}")
                df.loc[i, "File Location"] = series_uid

    df.to_csv(meta_path, index=False)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=str)
    parser.add_argument("meta_file_path", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("--retry", type=int, default=50)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.manifest_path), "data")
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    retry = args.retry

    while retry:
        download_result = nbia.downloadSeries(
            series_data=args.manifest_path,
            path=args.output_dir,
            input_type="manifest",
            csv_filename=args.meta_file_path,
            format="csv",
            as_zip=False,
        )

        if download_result is not None:
            break

        retry -= 1

    update_meta_filepath(args.output_dir, args.meta_file_path)
