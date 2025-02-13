import os
import requests
import zipfile
import shutil
import random
from tqdm import tqdm
import time
from PIL import Image

KITTI_CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']


def download_and_extract(url, destination, max_retries=5, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024  # 1 KB
            zip_path = os.path.join(destination, os.path.basename(url))

            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination)

            os.remove(zip_path)
            return True
        except requests.RequestException as e:
            print(f"An error occurred during the download: {e}")
            retries += 1
            print(f"Retrying in {retry_delay} seconds... ({retries}/{max_retries})")
            time.sleep(retry_delay)

    print("All download attempts failed.")
    return False


def convert_to_yolo_format(annotations, image_dir, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_id, ann_list in annotations.items():
        img_path = os.path.join(image_dir, f"{file_id}.png")  
        img = Image.open(img_path)
        width, height = img.size

        with open(os.path.join(output_path, f"{file_id}.txt"), 'w') as out_file:
            for ann in ann_list:
                classId, bbox = ann['classId'], ann['bbox']
                left, top, right, bottom = bbox

                x_center = (left + right) / 2.0 / width
                y_center = (top + bottom) / 2.0 / height
                bbox_width = (right - left) / width
                bbox_height = (bottom - top) / height

                yolo_bbox = [classId, x_center, y_center, bbox_width, bbox_height]
                yolo_bbox_str = " ".join(map(str, yolo_bbox))
                out_file.write(f"{yolo_bbox_str}\n")


def load_annotations(label_path):
    annotations = {}
    for ann_file in sorted(os.listdir(label_path)):
        if ann_file.endswith('.txt'):
            file_id = os.path.splitext(ann_file)[0]
            annotations[file_id] = []
            with open(os.path.join(label_path, ann_file), 'r') as file:
                for line in file.readlines():
                    elements = line.strip().split(' ')
                    object_type = elements[0]
                    if object_type not in KITTI_CLASS_NAMES:
                        continue
                    classId = KITTI_CLASS_NAMES.index(object_type)
                    bbox = list(map(float, elements[4:8]))  # left, top, right, bottom
                    annotations[file_id].append({
                        'classId': classId,
                        'bbox': bbox
                    })
    return annotations


def download_kitti(path, val_split=0.2):
    urls = {
        "data_object_image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "data_object_label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    }

    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(path, "training", "image_2"), exist_ok=True)
    os.makedirs(os.path.join(path, "training", "label_2"), exist_ok=True)

    # Download and extract each file sequentially
    for url in tqdm(urls.values(), desc="Downloading KITTI dataset"):
        if not download_and_extract(url, path):
            print(f"Download failed: {url}")
            return

    # Splitting into train and val directories
    image_files = os.listdir(os.path.join(path, "training", "image_2"))
    random.shuffle(image_files)
    val_count = int(len(image_files) * val_split)
    val_images = image_files[:val_count]
    train_images = image_files[val_count:]

    for img in tqdm(train_images, desc="Moving train images"):
        shutil.move(os.path.join(path, "training", "image_2", img), os.path.join(path, "images", "train", img))

    for img in tqdm(val_images, desc="Moving val images"):
        shutil.move(os.path.join(path, "training", "image_2", img), os.path.join(path, "images", "val", img))

    # Moving corresponding label files based on the images moved
    for img in tqdm(train_images, desc="Moving train labels"):
        lbl = img.replace(".png", ".txt")
        shutil.move(os.path.join(path, "training", "label_2", lbl), os.path.join(path, "labels", "train", lbl))

    for img in tqdm(val_images, desc="Moving val labels"):
        lbl = img.replace(".png", ".txt")
        shutil.move(os.path.join(path, "training", "label_2", lbl), os.path.join(path, "labels", "val", lbl))

    # Convert annotations to YOLO format
    train_annotations = load_annotations(os.path.join(path, "labels", "train"))
    val_annotations = load_annotations(os.path.join(path, "labels", "val"))

    convert_to_yolo_format(train_annotations, os.path.join(path, "images", "train"), os.path.join(path, "labels", "train"))
    convert_to_yolo_format(val_annotations, os.path.join(path, "images", "val"), os.path.join(path, "labels", "val"))

    shutil.rmtree(os.path.join(path, "training"))

    print("KITTI dataset downloaded, split into train and val sets.")


if __name__ == "__main__":
    download_kitti(path="../kitti")
