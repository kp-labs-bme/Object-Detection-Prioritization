# Smarter Sampling: Data Prioritization for Improved Object Detection

## Overview
This repository contains the implementation of our novel data prioritization method for object detection, as presented in our publication.

## Authors
- Csanád L. Balogh<sup>1</sup><sup>2</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=p0eQRMAAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0000-0001-9194-8574)
- Bence Szilárd Pap<sup>1</sup><sup>2</sup>
  - [Google Scholar](#)
  - [ORCID](#)
- Bálint Kővári<sup>1</sup><sup>2</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=WrtttXEAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0000-0003-2178-2921)
- Tamás Bécsi<sup>1</sup>
  - [Google Scholar](https://scholar.google.com/citations?user=Sdw_b5YAAAAJ&hl=hu&oi=ao)
  - [ORCID](https://orcid.org/0000-0002-1487-9672)

<sup>1</sup>Department of Control for Transportation and Vehicle Systems, Faculty of Transportation Engineering and Vehicle Engineering, Budapest University of Technology and Economics, H-1111 Budapest, Hungary

<sup>2</sup>Asura Technologies Ltd.

> **Abstract:**
>
> The effectiveness of deep learning models is strongly influenced by the quality of training data. Traditional training approaches assume that all samples contribute equally to the learning process, leading to uniform data sampling. However, this assumption does not account for the varying informational content of different samples. This paper presents a novel data prioritization method for object detection that dynamically adjusts the sampling probability of training data based on its relevance. Focusing on object detection applications within the computer vision field, the proposed methodology introduces the Relative Detection Error metric to evaluate and prioritize samples during training. By selecting data points with higher informational value, our approach improves both classification and localization accuracy while maintaining minimal computational overhead. The method is demonstrated using YOLO architectures on diverse datasets, showcasing its ability to generalize across different settings. Experimental results indicate that prioritizing high-value samples enhances the F1 score and mean Average Precision (mAP), leading to more efficient training and robust performance.

## Getting Started

### Running Locally

1. **Clone the Repository**
   ```sh
   git clone https://github.com/kp-labs-bme/Object-Detection-Prioritization
   cd Object-Detection-Prioritization
   ```
2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Download Datasets**
   Run the dataset downloader scripts to automatically download and structure the datasets:
   ```sh
   cd data/download/
   python download_coco.py
   python download_kitti.py
   python download_pascalvoc.py
   ```
4. **Train the Model**
   Run the training script from the root directory:
   ```sh
   python train.py
   ```
   Inputs can be provided via command line or set in `configuration.ini`.

### Running with Docker

1. **Navigate to the Docker Directory**
   ```sh
   cd docker
   ```
2. **Start Docker Container** (Pre-built script)
   ```sh
   ./start_docker.sh
   ```
3. **Alternatively, Manually Build and Run Docker**
   ```sh
   docker build -t object-detection .
   docker run -d --name object-detection-container object-detection
   docker exec -it object-detection-container /bin/bash
   ```

## Implementation Details

- The project builds upon **Ultralytics YOLO** as a base implementation.
- The **Relative Detection Error (RDE) metric** is introduced to dynamically prioritize data samples during training.
- The method is designed to generalize across multiple datasets, including COCO, KITTI, and Pascal VOC.

## Repository Structure
```
.
├───callbacks       # Callback functions for training
├───data           # Dataset handling
│   └───download  # Dataset downloader scripts (COCO, KITTI, Pascal VOC)
├───dataset        # Data loading and processing utilities
├───docker         # Docker setup scripts
├───factory        # Factory design pattern for modular components
├───models         # Model weights are downloaded here
├───static         # Static files for github pages       
├───trainer        # Training-related code
└───utils          # General utility functions
```


## Citation

If you use this work, please cite it as follows:

```bibtex
@article{,
  author = {Csanád L. Balogh, Bence Szilárd Pap, Bálint Kővári, Tamás Bécsi},
  title = {Smarter Sampling: Data Prioritization for Improved Object Detection},
  journal = {},
  year = {2025},
  doi = {}
}
```
