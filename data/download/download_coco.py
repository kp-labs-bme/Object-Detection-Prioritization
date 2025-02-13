import os
import requests
import zipfile
import shutil
import random
from tqdm import tqdm
import time
from pycocotools.coco import COCO
from PIL import Image

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

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


def convert_to_yolo_format(coco, image_ids, output_path, image_dir):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = Image.open(img_path)
        width, height = img.size

        label_file_path = os.path.join(output_path, f"{os.path.splitext(img_info['file_name'])[0]}.txt")
        with open(label_file_path, 'w') as out_file:
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
                class_id = COCO_CLASS_NAMES.index(coco.loadCats(ann['category_id'])[0]['name'])
                bbox = ann['bbox']

                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                bbox_width = bbox[2] / width
                bbox_height = bbox[3] / height

                yolo_bbox = [class_id, x_center, y_center, bbox_width, bbox_height]
                yolo_bbox_str = " ".join(map(str, yolo_bbox))
                out_file.write(f"{yolo_bbox_str}\n")


def download_coco(path):
    urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }

    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "val"), exist_ok=True)

    # Download and extract each file sequentially
    for url in tqdm(urls.values(), desc="Downloading COCO dataset"):
        if not download_and_extract(url, path):
            print(f"Download failed: {url}")
            return

    # Load COCO annotations
    coco_train = COCO(os.path.join(path, "annotations", "instances_train2017.json"))
    coco_val = COCO(os.path.join(path, "annotations", "instances_val2017.json"))

    # Get image IDs
    train_img_ids = coco_train.getImgIds()
    val_img_ids = coco_val.getImgIds()

    # Convert annotations to YOLO format
    convert_to_yolo_format(coco_train, train_img_ids, os.path.join(path, "labels", "train"), os.path.join(path, "train2017"))
    convert_to_yolo_format(coco_val, val_img_ids, os.path.join(path, "labels", "val"), os.path.join(path, "val2017"))

    # Move corresponding images
    for img_id in tqdm(train_img_ids, desc="Moving train images"):
        img_info = coco_train.loadImgs(img_id)[0]
        shutil.move(os.path.join(path, "train2017", img_info['file_name']), os.path.join(path, "images", "train", img_info['file_name']))

    for img_id in tqdm(val_img_ids, desc="Moving val images"):
        img_info = coco_val.loadImgs(img_id)[0]
        shutil.move(os.path.join(path, "val2017", img_info['file_name']), os.path.join(path, "images", "val", img_info['file_name']))

    shutil.rmtree(os.path.join(path, "train2017"))
    shutil.rmtree(os.path.join(path, "val2017"))

    print("COCO dataset downloaded, split into train and val sets.")

if __name__ == "__main__":
    download_coco(path="../coco")
