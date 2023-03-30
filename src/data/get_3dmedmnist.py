import argparse
import csv
from pathlib import Path

import numpy as np
from medmnist.dataset import MedMNIST3D


class MedMNISTWrapper(MedMNIST3D):
    "Wrapper class makes it possible to specify the flag."

    def __init__(self, flag, split, root, download=True):
        self.flag = flag
        super().__init__(split=split, download=download, root=root)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )

    args = parser.parse_args()
    return args


def download_data(data_root):
    splits_dir = Path(data_root).parent / "data_splits"
    Path(data_root).mkdir(exist_ok=True)
    splits_dir.mkdir(exist_ok=True, parents=True)
    for task in [
        "organmnist3d",
        "nodulemnist3d",
        "fracturemnist3d",
        "adrenalmnist3d",
        "vesselmnist3d",
        "synapsemnist3d",
    ]:

        for split in ["train", "val", "test"]:
            dataset = MedMNISTWrapper(split=split, flag=task, root=data_root)
            data = dataset.imgs.shape
            # loop over each volume and save as npy
            out_dir = Path(data_root) / task / split
            out_dir.mkdir(exist_ok=True, parents=True)
            dataset_list = []
            for i in range(data[0]):
                np.save(
                    out_dir / f"{i}.npy",
                    dataset.imgs[i],
                )

                dataset_list.append(out_dir / f"{i}.npy")
            print(f"{task} {split} with {len(dataset_list)} images")
            save_list_as_csv(
                dataset_list,
                splits_dir
                / f"medmnist3d_{task}_{split.replace('ing','').replace('idation','')}.csv",
            )


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


if __name__ == "__main__":
    args = parse_args()
    download_data(data_root=args.data_root)
