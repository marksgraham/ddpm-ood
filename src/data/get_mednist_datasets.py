import argparse
import csv
from pathlib import Path

from monai.apps import MedNISTDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )

    args = parser.parse_args()
    return args


def download_data(data_root):
    splits_dir = Path(data_root) / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)
    for split in ["training", "validation", "test"]:
        dataset = MedNISTDataset(
            root_dir=args.data_root, section=split, download=True, progress=False, seed=0
        )
        data_list = dataset.data
        datasets = set(item["class_name"] for item in data_list)
        for dataset in datasets:
            dataset_list = [item["image"] for item in data_list if item["class_name"] == dataset]
            save_list_as_csv(
                dataset_list,
                splits_dir / f"{dataset}_{split.replace('ing','').replace('idation','')}.csv",
            )
        print("debug")


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


if __name__ == "__main__":
    args = parse_args()
    download_data(data_root=args.data_root)
