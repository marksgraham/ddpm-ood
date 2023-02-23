import argparse
import csv
from pathlib import Path

from monai.apps import DecathlonDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )

    args = parser.parse_args()
    return args


def download_data(data_root):
    splits_dir = Path(data_root).parent / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)
    for task in [
        "Task01_BrainTumour",
        "Task02_Heart",
        "Task03_Liver",
        "Task04_Hippocampus",
        "Task05_Prostate",
        "Task06_Lung",
        "Task07_Pancreas",
        "Task08_HepaticVessel",
        "Task09_Spleen",
        "Task10_Colon",
    ]:

        for split in ["training", "validation", "test"]:
            dataset = DecathlonDataset(
                root_dir=args.data_root,
                task=task,
                section=split,
                download=True,
                progress=True,
                seed=0,
                cache_rate=0,
            )
            data_list = dataset.data
            dataset_list = [item["image"] for item in data_list]
            print(f"{task} {split} with {len(dataset_list)} images")
            save_list_as_csv(
                dataset_list,
                splits_dir / f"{task}_{split.replace('ing','').replace('idation','')}.csv",
            )


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


if __name__ == "__main__":
    args = parse_args()
    download_data(data_root=args.data_root)
