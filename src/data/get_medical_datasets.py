import argparse
import csv
from pathlib import Path

from monai.apps import MedNISTDataset
from sklearn.model_selection import train_test_split


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


def create_train_test_splits(data_root):
    splits_dir = Path(data_root) / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    # need to create a train/val split for these datasets
    for dataset in ["FashionMNIST", "MNIST", "CIFAR10", "SVHN"]:
        numpy_data_root = Path(data_root) / dataset / "numpy"
        train_and_val_list = list((numpy_data_root / "train").glob("*"))
        train_list, val_list = train_test_split(train_and_val_list, test_size=0.05, random_state=42)
        test_list = list((numpy_data_root / "test").glob("*"))
        for split_name, data_split in zip(
            ["train", "val", "test"], [train_list, val_list, test_list]
        ):
            save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")

    # CelebA already has a train/val split
    dataset = "CelebA"
    numpy_data_root = Path(data_root) / dataset / "numpy"
    train_list = list((numpy_data_root / "train").glob("*"))
    val_list = list((numpy_data_root / "valid").glob("*"))
    test_list = list((numpy_data_root / "test").glob("*"))
    for split_name, data_split in zip(["train", "val", "test"], [train_list, val_list, test_list]):
        save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")


if __name__ == "__main__":
    args = parse_args()
    download_data(data_root=args.data_root)
    create_train_test_splits(data_root=args.data_root)
