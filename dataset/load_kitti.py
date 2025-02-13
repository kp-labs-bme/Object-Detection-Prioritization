import os
from utils.augmentation import setup_augmentation
from data.download.download_kitti import download_kitti


def load_kitti(config, data_dir, val_split=0.2) -> dict:
    train_data_path = os.path.join(data_dir, "images", "train")
    val_data_path = os.path.join(data_dir, "images", "val")
    train_ann_path = os.path.join(data_dir, "labels", "train")
    val_ann_path = os.path.join(data_dir, "labels", "val")

    # Ellenőrizd, hogy a fájlok léteznek-e
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path) or \
       not os.path.exists(train_ann_path) or not os.path.exists(val_ann_path):
        print("KITTI dataset not found, downloading...")
        download_kitti(data_dir, val_split)

    # Ellenőrizd újra, hogy a fájlok léteznek-e a letöltés után
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path) or \
       not os.path.exists(train_ann_path) or not os.path.exists(val_ann_path):
        raise FileNotFoundError("KITTI dataset could not be downloaded or is missing files.")

    train_transforms, val_transforms = setup_augmentation(config)

    kitti_data = dict()
    kitti_data["train_data_path"] = train_data_path
    kitti_data["val_data_path"] = val_data_path
    kitti_data["train_ann_path"] = train_ann_path
    kitti_data["val_ann_path"] = val_ann_path
    kitti_data["train_transforms"] = train_transforms
    kitti_data["val_transforms"] = val_transforms

    return kitti_data
