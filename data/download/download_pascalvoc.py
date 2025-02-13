import os
import requests
import tarfile
import shutil
import random
from tqdm import tqdm
import time
from PIL import Image
import xml.etree.ElementTree as ET

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

missing_images = []  # Store missing image paths
missing_annotations = []  # Store missing annotation paths


def download_and_extract(url, destination, max_retries=5, retry_delay=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024  # 1 KB
            tar_path = os.path.join(destination, os.path.basename(url))

            print("Downloading dataset...")
            with open(tar_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

            print("Extracting dataset...")
            if tarfile.is_tarfile(tar_path):
                with tarfile.open(tar_path) as tar_ref:
                    tar_ref.extractall(destination)
            else:
                print("Downloaded file is not a tar file.")

            os.remove(tar_path)  # Remove the tar file after extraction
            return True
        except requests.RequestException as e:
            print(f"An error occurred during the download: {e}")
            retries += 1
            print(f"Retrying in {retry_delay} seconds... ({retries}/{max_retries})")
            time.sleep(retry_delay)

    print("All download attempts failed.")
    return False


def convert_voc_to_yolo_format(annotation_path, image_size):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in VOC_CLASS_NAMES:
            continue
        class_id = VOC_CLASS_NAMES.index(class_name)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2.0 / image_size[0]
        y_center = (ymin + ymax) / 2.0 / image_size[1]
        bbox_width = (xmax - xmin) / image_size[0]
        bbox_height = (ymax - ymin) / image_size[1]

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

    return yolo_annotations


def load_and_convert_voc_annotations(label_dir, image_dir, output_dir):
    global missing_images, missing_annotations

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = sorted(os.listdir(image_dir))
    annotation_list = sorted(os.listdir(label_dir))

    # Create a set of image and annotation file names (without extensions) for comparison
    image_files = {os.path.splitext(img)[0] for img in image_list}
    annotation_files = {os.path.splitext(ann)[0] for ann in annotation_list}

    # Only check for annotations and images in the current split
    matched_annotations = image_files & annotation_files  # Only process images that have annotations in this split
    unmatched_images = image_files - annotation_files  # Images without annotations
    unmatched_annotations = annotation_files - image_files  # Annotations without images

    # Collect missing annotations
    missing_annotations_in_split = list(unmatched_annotations)

    # Process only the matched image-annotation pairs
    for file_id in matched_annotations:
        img_path = os.path.join(image_dir, f"{file_id}.jpg")
        ann_path = os.path.join(label_dir, f"{file_id}.xml")

        if not os.path.exists(img_path):
            missing_images.append(img_path)
            continue

        img = Image.open(img_path)
        width, height = img.size

        yolo_annotations = convert_voc_to_yolo_format(ann_path, (width, height))

        with open(os.path.join(output_dir, f"{file_id}.txt"), 'w') as out_file:
            out_file.write("\n".join(yolo_annotations))

    return missing_annotations_in_split  # Return the missing annotations for this split


def download_voc(path, val_split=0.2):
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(path, "labels", "val"), exist_ok=True)

    # Download and extract VOC dataset
    if not download_and_extract(url, path):
        print(f"Download failed: {url}")
        return

    voc_dir = os.path.join(path, "VOCdevkit", "VOC2012")

    total_images = len(os.listdir(os.path.join(voc_dir, 'JPEGImages')))
    total_annotations = len(os.listdir(os.path.join(voc_dir, 'Annotations')))
    print(f"Total images in VOC2012/JPEGImages: {total_images}")
    print(f"Total annotations in VOC2012/Annotations: {total_annotations}")

    image_files = os.listdir(os.path.join(voc_dir, "JPEGImages"))
    random.shuffle(image_files)
    val_count = int(len(image_files) * val_split)
    val_images = image_files[:val_count]
    train_images = image_files[val_count:]

    print(f"Splitting into {len(train_images)} training images and {len(val_images)} validation images.")

    # Move train images
    for img in tqdm(train_images, desc="Moving train images"):
        shutil.copy(os.path.join(voc_dir, "JPEGImages", img), os.path.join(path, "images", "train", img))

    # Move val images
    for img in tqdm(val_images, desc="Moving val images"):
        shutil.copy(os.path.join(voc_dir, "JPEGImages", img), os.path.join(path, "images", "val", img))

    print(f"Total train images moved: {len(os.listdir(os.path.join(path, 'images', 'train')))}")
    print(f"Total val images moved: {len(os.listdir(os.path.join(path, 'images', 'val')))}")

    # Convert train labels and get missing annotations
    missing_train_annotations = load_and_convert_voc_annotations(
        os.path.join(voc_dir, "Annotations"), 
        os.path.join(path, "images", "train"),
        os.path.join(path, "labels", "train")
    )

    # Convert val labels and get missing annotations
    missing_val_annotations = load_and_convert_voc_annotations(
        os.path.join(voc_dir, "Annotations"), 
        os.path.join(path, "images", "val"),
        os.path.join(path, "labels", "val")
    )

    # Cross-check missing annotations between train and val
    missing_annotations_in_both = set(missing_train_annotations) & set(missing_val_annotations)

    if missing_annotations_in_both:
        print(f"Warning: {len(missing_annotations_in_both)} annotations are missing in both train and val splits.")

    shutil.rmtree(os.path.join(path, "VOCdevkit"))

    # Summary of missing files
    if missing_images:
        print(f"\nTotal missing images: {len(missing_images)}")
    if missing_train_annotations or missing_val_annotations:
        print(f"Total missing annotations in train: {len(missing_train_annotations)}")
        print(f"Total missing annotations in val: {len(missing_val_annotations)}")
        print(f"Total annotations missing from both train and val: {len(missing_annotations_in_both)}")

    print("Pascal VOC 2012 dataset downloaded and converted.")

if __name__ == "__main__":
    download_voc(path="../pascal")
