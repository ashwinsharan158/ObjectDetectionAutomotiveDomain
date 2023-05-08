# Object Detection in Automotive Domain: A Supervised Learning Comparison

## Abstract

 This scientific paper proposes the development and comparison of object detection models for detecting cars, pedestrians, and street signs in the automotive domain. The study aims to train and assess the performance of each model using supervised learning approaches, including YOLO, faster R-CNN, and SVM models. The gathered information will be tagged with bounding boxes or masks to locate the objects in the image. The primary objective of the project is to efficiently detect and classify objects using various supervised learning approaches to address the problem of object detection.

[//]: # (Image References)

[image1]: ./output_images/img1.PNG "Car Samples"
[image2]: ./output_images/img2.PNG "Non-car Samples"
[image3]: ./output_images/img3.PNG "HOG Comparison"
[image4]: ./output_images/img4.PNG "HOG Comparison"

## Methodology

The following steps are taken to train YOLOv8 and ResNet-50-FPN models 
* Data Collection: The COCO 2017 Train dataset's car, person, and stop sign classes [1] are combined with the INRIA person dataset [5] to create a custom dataset for the YOLO and Faster R-CNN models. This custom dataset contains 7470 images in total, with 9611 instances of vehicles, 11131 instances of persons, and 1983 instances of stop signs.

* Data Conversion: Annotations are converted to Darknet TXT format for the YOLOv8 model and VOC XML format for the Faster R-CNN ResNet-50-FPN model. This step ensures that the data is in a format that is compatible with the respective models.

* Data Shuffling: The data is shuffled to increase the accuracy of the models. This step is essential to avoid any bias in the data and ensure that the models can generalize well.

* Data Splitting: The shuffled data is then split into three parts: training, validation, and testing. 80% of the data is selected for training, 10% for validation, and 10% for testing. This split ensures that the models are trained on a significant amount of data, and their performance is evaluated on previously unseen data.

* Data Preprocessing: Before training the models, the images are normalized and resized to 640x640 in the models' configuration phase. The bounding box values are also modified accordingly to ensure that the objects' locations are accurately represented in the models.

* Model Training: The YOLOv8 and Faster R-CNN ResNet-50-FPN models are trained on the custom dataset using the training set. The models are trained to detect and classify vehicles, persons, and stop signs in the images.

* Model Validation: The validation set is used to evaluate the models' performance and fine-tune their hyperparameters to improve their accuracy.


* Model Testing: The testing set is used to evaluate the models' final performance and assess their ability to generalize to previously unseen data.

* Visual Evaluation: The models are run on pre recorded video of street traffic as shown in the figures below.

[![Resnet50 visualization](http://img.youtube.com/vi/EOq661_DQ4I/0.jpg)](https://youtu.be/EOq661_DQ4I "Resnet50 visualization")

<b>Click to play Resnet50 visualization.</b>


[![YOLO visualization](http://img.youtube.com/vi/SnlIH8-2X1Q/0.jpg)](https://youtu.be/SnlIH8-2X1Q "YOLO visualization")

<b>Click to play YOLO visualization</b>

The following are the steps taken to train the SVM model.
* Data collection: The COCO 2017 Train dataset's person and stop sign classes[1]  are cropped according to their bounding box annotation size and resized to 64 x 64. Also detailed car images were taken from the KITTI vision benchmark suite [6]. The dataset contains 8792 images of cars, 11237 images of people, 1983 images of stop-sign and 16171 non-domain images.

* Feature Extraction: The Histogram of Oriented Gradients are extracted for each individual domain and the non domain Image as seen in the visualizations below. To classify more accurately, histogram and binned color features are also extracted.

* Data preparation: Each domain image feature is stacked with non-domain features given an output label of 1 for domain and 0 for non-domain. Then split into 80% for training and 20% for testing.  

* Classification: Linear SVC to train 3 separate models on the data to classify cars, person and stop sign in images.

* Sliding window: HOG window search routine is used, where it scans select portions of the image and checks for presence of a domain object.

* Temporal heat mapping: Heat mapping of successful hits in the frame to reduce false positives and improve accuracy.

* Visual Evaluation: The models are run on pre recorded video of street traffic as shown in the figures below.

## Evaluation 
SVM:
In the scope of this study, the SVM model was found to be insufficient in detecting pedestrians and stop signs. However, it performed well in detecting vehicles due to the availability of a large number of diverse images from the KITTI dataset [6] with various angles and better compositions. Additionally, a vast collection of non-domain images of empty roads enabled the model to differentiate between the background and vehicles effectively.

### YOLO:
The model was found to be suitable in accurately detecting vehicles, pedestrians and stop signs with a speed of 45fps in real-time video.

In terms of accuracy:

![alt_text][image3]

### Faster R-CNN:
The model was found to be suitable in accurately detecting vehicles, pedestrians and stop signs with a speed of 15fps in real-time video.

In terms of accuracy:

![alt_text][image4]


# Conclusion 
In conclusion, this scientific paper presented the development and comparison of object detection models for detecting cars, pedestrians, and street signs in the automotive domain. Three supervised learning approaches were implemented: YOLO, faster R-CNN, and SVM models. The models were trained and evaluated using the Coco [1] and KITTI [6] datasets, with the gathered information tagged with bounding boxes or masks to locate the objects in the image. 

Our experimental results showed that YOLO outperformed the other models with a 0.90 mean average precision (MAP) and a speed of 40 frames per second (fps). Faster R-CNN and SVM models showed lower performance in terms of both accuracy and speed. Therefore, we recommend the use of YOLO for object detection tasks in the automotive domain


