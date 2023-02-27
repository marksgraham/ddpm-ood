import argparse
import csv
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )

    args = parser.parse_args()
    return args


def prepare_data(data_root):
    # # cromis for train and val
    # data_dir = Path(args.data_root) / 'cromis'
    # # symlink dirs
    # for f in ['train', 'val']:
    #     data_target = data_dir / f
    #     data_target.mkdir(exist_ok=True, parents=True)
    #     source_files = list(Path(f'/home/mark/data_drive/generative_ct/data/nonrigid_cromis_all/images/{f}/').glob('*.nii*'))
    #     for s in source_files:
    #         os.symlink(s, data_target / s.name)
    # for f in ['train', 'val']:
    #     # create data splits
    #     data_target = data_dir / f
    #     image_list = list(data_target.glob('*.nii*'))
    #     save_list_as_csv(image_list, Path(args.data_root) / f'data_splits/cromis_{f}.csv')

    # yee for test
    # data_dir = Path(args.data_root) / 'yee'
    # # symlink dirs
    # for f in ['test']:
    #     data_target = data_dir / f
    #     data_target.mkdir(exist_ok=True, parents=True)
    #     source_files = list(Path(f'/home/mark/data_drive/generative_ct/data/nonrigid_yee/images/val/ims/').glob('*.nii*'))
    #     for s in source_files:
    #         os.symlink(s, data_target / s.name)
    # for f in ['test']:
    #     # create data splits
    #     data_target = data_dir / f
    #     image_list = list(data_target.glob('*.nii*'))
    #     save_list_as_csv(image_list, Path(args.data_root) / f'data_splits/yee_{f}.csv')

    # yee corrupted for ood
    data_dir = Path(args.data_root) / "yee_corrupted"
    # symlink dirs
    for f in ["test"]:
        data_target = data_dir / f
        data_target.mkdir(exist_ok=True, parents=True)
        source_files = list(
            Path("/home/mark/data_drive/generative_ct/data/nonrigid_yee/corrupted/images/").glob(
                "*.nii*"
            )
        )
        for s in source_files:
            os.symlink(s, data_target / s.name)
    for f in ["test"]:
        # create data splits
        data_target = data_dir / f
        image_list = list(data_target.glob("*.nii*"))
        save_list_as_csv(image_list, Path(args.data_root) / f"data_splits/yee_corrupted_{f}.csv")

    # decathlon for ood
    data_dir = Path(args.data_root) / "decathlon"
    # symlink dirs
    for f in ["test"]:
        data_target = data_dir / f
        data_target.mkdir(exist_ok=True, parents=True)
        source_files = list(
            Path(
                "/home/mark/data_drive/generative_ct/data/medical-decathlon-test-medium/images/val/"
            ).glob("*.nii*")
        )
        for s in source_files:
            os.symlink(s, data_target / s.name)
    for f in ["test"]:
        # create data splits
        data_target = data_dir / f
        image_list = list(data_target.glob("*.nii*"))
        save_list_as_csv(image_list, Path(args.data_root) / f"data_splits/decathlon_{f}.csv")


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


if __name__ == "__main__":
    args = parse_args()
    prepare_data(data_root=args.data_root)
