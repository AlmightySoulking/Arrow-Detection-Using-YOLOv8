# Arrow-Detection-Using-YOLOv8

Arrow Detection with YOLOv8
This project utilizes the YOLOv8 object detection model to identify and classify arrows pointing either left or right within images. It serves as a practical example of training a state-of-the-art deep learning model on a custom dataset.

📖 Table of Contents
Project Description

Dataset

Getting Started

Prerequisites

Installation

Usage

Training the Model

Running Predictions

Results

Contributing

License

📝 Project Description
The core of this project is a Jupyter Notebook (arrowDetection.ipynb) that provides a step-by-step guide through the entire machine learning workflow. This includes:

Data Setup: Preparing the dataset and creating the necessary configuration files for YOLOv8.

Model Training: Fine-tuning a pre-trained YOLOv8s model on the custom arrow dataset.

Inference: Using the trained model to perform predictions on new images and evaluating its performance.

🖼️ Dataset
This project uses the "ArrOW.v1i.yolov8" dataset, which contains images of arrows annotated with bounding boxes for "left" and "right" classes.

A key part of the notebook is a helper function that automatically generates the data.yaml file. This configuration file is crucial for YOLOv8, as it defines:

The path to the training and validation image sets.

The total number of classes.

The names of the classes.

The generated data.yaml will look like this:

path: ''
train: ArrOW.v1i.yolov8/train
val: ArrOW.v1i.yolov8/valid
nc: 2
names:
- left
- right

🚀 Getting Started
Follow these instructions to get a local copy of the project up and running.

Prerequisites
Ensure you have Python 3 installed on your system. The primary dependencies are:

ultralytics (for the YOLOv8 framework)

pyyaml (for handling YAML configuration files)

Installation
Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install the required Python packages:

pip install ultralytics pyyaml

⚙️ Usage
The Jupyter Notebook is the main entry point for interacting with the project.

Training the Model
To start training your own arrow detection model, execute the following command within the notebook:

!yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=150

model=yolov8s.pt: Starts with the small, pre-trained YOLOv8 model.

data=data.yaml: Points to our dataset configuration.

epochs=100: The model will be trained for 100 complete passes over the dataset.

imgsz=150: All images will be resized to 150x150 pixels.

Running Predictions
After training, you can use the best-performing model checkpoint to make predictions on new images:

!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=ArrOW.v1i.yolov8/valid/images save=True

model=.../best.pt: Specifies the path to the best weights saved during training.

source=.../images: The directory of images you want to test the model on.

save=True: The prediction results (images with bounding boxes) will be saved to a directory.

📊 Results
After 100 epochs of training, the model achieved the following performance metrics on the validation set:

Metric

Value

mAP50

0.635

mAP50-95

0.468

The model demonstrates a solid ability to detect both "left" and "right" arrows with respectable precision and recall, proving the effectiveness of YOLOv8 for this custom detection task.

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is distributed under the MIT License. See the LICENSE file for more information.
