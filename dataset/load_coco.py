import os
import requests
import zipfile
from pycocotools.coco import COCO
import yaml
import shutil


def download_coco():
    coco_url = 'http://images.cocodataset.org/zips/val2017.zip'
    ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    # Get the directory of the script file (the directory where the script is located)
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set the data directory one level higher in a 'data/coco' folder relative to the script location
    data_dir = os.path.abspath(os.path.join(script_directory, '..', 'data', 'coco'))

    print(f"Setting dataset folder to {data_dir}")

    img_dir = os.path.join(data_dir, 'images')
    ann_dir = os.path.join(data_dir, 'annotations')

    if os.path.exists(data_dir):
        print(f"Folder already exists at {data_dir}")
    else:
        os.makedirs(data_dir)
        print(f"Created folder at {data_dir}")

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)

    img_zip = os.path.join(data_dir, 'val2017.zip')
    ann_zip = os.path.join(data_dir, 'annotations_trainval2017.zip')

    if not os.path.exists(os.path.join(img_dir, 'val2017')):
        print(f'Downloading COCO images to {img_dir}')
        download_file(coco_url, img_zip)
        with zipfile.ZipFile(img_zip, 'r') as zip_ref:
            zip_ref.extractall(img_dir)
    else:
        print(f"Images already exist at {img_dir}")

    if not os.path.exists(os.path.join(ann_dir, 'instances_val2017.json')):
        print(f'Downloading COCO annotations to {ann_dir}')
        download_file(ann_url, ann_zip)
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if filename:  # skip directories
                    source = zip_ref.open(member)
                    target = open(os.path.join(ann_dir, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
    else:
        print(f"Annotations already exist at {ann_dir}")

    # Verify that the expected files are present
    if not os.path.exists(os.path.join(img_dir, 'val2017')):
        raise FileNotFoundError(f'Expected directory {os.path.join(img_dir, "val2017")} not found.')

    if not os.path.exists(os.path.join(ann_dir, 'instances_val2017.json')):
        print(f"Files found in annotation directory: {os.listdir(ann_dir)}")
        raise FileNotFoundError(f'Expected file {os.path.join(ann_dir, "instances_val2017.json")} not found.')


def download_file(url, dest):
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)


def convert_coco_to_yolo():
    # Get the directory of the script file (the directory where the script is located)
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set the data directory one level higher in a 'data/coco' folder relative to the script location
    data_dir = os.path.abspath(os.path.join(script_directory, '..', 'data', 'coco'))

    img_dir = os.path.join(data_dir, 'images', 'val2017')
    ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    labels_dir = os.path.join(data_dir, 'labels')

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f'Annotation file {ann_file} not found.')

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        label_path = os.path.join(labels_dir, f'{os.path.splitext(img_info["file_name"])[0]}.txt')
        with open(label_path, 'w') as label_file:
            for ann in anns:
                bbox = ann['bbox']
                # Convert COCO bbox to YOLO format
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                width = bbox[2]
                height = bbox[3]

                img_width = img_info['width']
                img_height = img_info['height']

                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height

                category_id = ann['category_id']
                label_file.write(f'{category_id} {x_center} {y_center} {width} {height}\n')


def create_yolo_data_dict():
    # Get the directory of the script file (the directory where the script is located)
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set the data directory one level higher in a 'data/coco' folder relative to the script location
    data_dir = os.path.abspath(os.path.join(script_directory, '..', 'data', 'coco'))

    img_dir = os.path.join(data_dir, 'images', 'val2017')
    labels_dir = os.path.join(data_dir, 'labels')
    train_transforms = None  # Define your transformations here
    val_transforms = None  # Define your transformations here

    yolo_data = dict()
    yolo_data["img_dir"] = img_dir
    yolo_data["labels_dir"] = labels_dir
    yolo_data["train_transforms"] = train_transforms
    yolo_data["val_transforms"] = val_transforms
    yolo_data["dataset_config"] = create_yolo_dataset_yaml(data_dir)

    return yolo_data


def create_yolo_dataset_yaml(data_dir):
    dataset_yaml = {
        'train': os.path.join(data_dir, 'images', 'train2017'),  # Modify this path accordingly
        'val': os.path.join(data_dir, 'images', 'val2017'),
        'test': os.path.join(data_dir, 'images', 'test2017'),  # Modify this path accordingly if test data is available
        'nc': 80,  # Number of classes in COCO dataset
        'names': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
            'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    }

    dataset_yaml_path = os.path.join(data_dir, 'coco.yaml')
    with open(dataset_yaml_path, 'w') as file:
        yaml.dump(dataset_yaml, file)

    return dataset_yaml_path


# def load_coco():
#     download_coco()
#     convert_coco_to_yolo()
#     yolo_data = create_yolo_data_dict()
#     print("YOLO data dictionary created successfully.")
#     return yolo_data


if __name__ == "__main__":
    download_coco()
    convert_coco_to_yolo()
    yolo_data = create_yolo_data_dict()
    print("YOLO data dictionary created successfully.")
