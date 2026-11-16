---
title: Homework 1 -- Your Title Here
author: John Doe
course: Artificial Intelligence · UCU
date: 2026-10-12
---

### EDA:

We've started our work with EDA to get to know our data and prepare it for our models.
We've found:
1. Significant class imbalance: cars(44.0%) and people(31.1%) together take more than 75% of the dataset, while buses are only 0.8 %. This may cause models to overfit to predicting only dominant classes and discard all other ones. For now we decided to not use any data augmentation to see how the models will perform without them, but we will include those later if we see the need for those.
2. Below you can see sizes distribution across different object types and clips. We can see that those wary highly, but the biggest part of object is at around 32 to 96 px.
![Sizes distribution](assets/sizes_distribution.png)
3. The frames are mostly very densely covered in objects with average of 45.1% object per clip, which will be a challenge to models as they will have to be able to predict overlapping bounding boxes.
4. We converted COCO annotations into YOLO format for our models. To validate the success we sampled a few images and displayed them with ground truth boxes.
5. We didn't resize or normalize the frames during EDA step, this happens in later steps to be able to experiment with different sizes if our chosen size is too small to be able to detect smaller objects. Both resizing and validation happens inside the torchvision model, where images are resized to passed size and normalized to floats [0.0, 1.0]
6. To handle ignore regions we painted them gray during training part and ignored detections inside them while calculating scores on validation.
7. We've split the dataset into train and validate by clip with fixed seed 42. The validation dataset was chosen based on how close the percentages of different classes is to the percentages in the whole dataset. Our train data contains 55 clips and validation contains 8, which is not much and will be also used to choose best parameters for models, so we will also rely on leaderboard score for our decision.
8. As consecutive frames are very similar to each other we decided not to use all of them. To choose subsampling step we calculated persistence and mean and median IoU. Persistence remained very high even for subsampling step 10, so our main metrics was mean IoU. We chose step = 3, as it was the highest one where IoU stayed higher than 0.6, so we removed redundancy and lowered training time while still being able to see object trajectories across frames.

### Two-stage anchor based (FasterCNN)

We've decided to use Faster CNN for this task because it is faster and more lightweight then Cascade R-CNN but still accurate. We used the pretrained weights from FasterRCNN_ResNet50_FPN_Weights.COCO_V1. We chose them because they are pretrained on COCO dataset, which contains object of the same classes we are working with. This will both lower the risk of overfitting on our relatively small dataset and speed up the training process.

This is the config of our initial model:

 '# Training setup: 
 Faster R-CNN
 Model: torchvision Faster R-CNN, ResNet-50-FPN (v1), initialised from COCO weights `FasterRCNN_ResNet50_FPN_Weights.COCO_V1`; 
 box predictor replaced by a new 9-way head (8 classes + background)
 Anchors: sizes per FPN level ((16,), (32,), (64,), (128,), (256,)), aspect ratios (0.5, 1, 2)
 RPN: 1000 proposals after NMS (train and test); RoI head samples 512 RoIs/image, positive fraction 0.25
 Input: long side resized to 1536 px (aspect ratio kept); torchvision normalises with ImageNet mean/std
 Backbone: frozen BatchNorm, last 3 ResNet stages + FPN + heads trained
 Inference: score threshold 0.001, NMS IoU 0.5, max 500 detections/image
 Optimiser: SGD momentum 0.9, lr 0.005, weight decay 0.0001, cosine schedule, 300 warm-up iterations, grad-clip 10.0
 Batch size 4, 3 epochs, mixed precision: True
 Augmentation: horizontal flip p=0.5, colour jitter 0.2 (brightness/contrast/saturation, hue/4), random zoom-in crop p=0.0 (side fraction 0.5-1); ignored regions are masked in the train images
 Model selection: last epoch

We've decided to investigate two parameters:
1. Number of rpn proposals, which controls how many of candidate regions are passed to second stage. As our frames are very densely filled with objects, we wanted to make sure that too low number of proposals won't pull our performance down.
2. Box positive fraction, which controls what portion of all proposals must be positive and what portion should be background. Because our frames contain a lot of objects many of which are overlapping we wanted to make sure our model sees enough real objects and learns them.

To tune those parameters we first established the baseline, which used rpn_proposals = 1000 and box_positive_fraction = 0.25.

Then we under the same conditions changed number of rpn proposals to 300 and to 2000 and then box positive fraction to 0.10 and to 0.50.

All the runs did 3 epochs on every second prepared frame. 

Number of rpn proposals didn't change mAP much (0.120 at 1000, 0.117 at 300 and 0.116 at 2000), so we sticked with the default value of 1000.

Box positive fraction of 0.5 raised the result but only slightly (from intial 0.120 to 0.122) but when we lowered it to 0.10 mAP dropped to 0.106 because second stage trained mostly on background and wasn't able to classify objects accurately.

![Loss plots](./images/frcnn_losses.png)
On the graphs above we can see that loss objectness and loss rpn box regression don't change as we change the parameters because we didn't tune any parameters which affect those.

But they did affect classifier loss and box regression loss. Box positive fraction of 0.10 differs the most significantly because second stage focuses on background and learns it quickly so loss is small. But it doesn't learn actual objects so the results are the lowest.

| experiment | rpn proposals | box positive fraction | mAP | AP50 | AP small | AP medium | AP_large | ms per img | fps | train_min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1000 | 0.25 | 0.120 | 0.276 | 0.039 | 0.104 | 0.250 | 24.241 | 41.253 | 52.469 |
| rpn300 | 300 | 0.25 | 0.117 | 0.267 | 0.031 | 0.099 | 0.251 | 23.847 | 41.933 | 46.785 |
| rpn 2000 | 2000 | 0.25 | 0.116 | 0.270 | 0.036 | 0.101 | 0.248 | 29.511 | 33.886 | 47.150 |
| pos 0.10 | 1000 | 0.10 | 0.106 | 0.250 | 0.033 | 0.095 | 0.226 | 23.555 | 42.454 | 48.297 |
| pos 0.50 | 1000 | 0.50 | 0.122 | 0.282 | 0.035 | 0.106 | 0.255 | 26.846 | 37.249 | 47.666 |


#### Main failures:

We can see that the smaller the object the harder it is for model to detect and classify because the less pixels the object takes the easier it is for model to completely miss it or misclassify it. We will try fixing it by increasing the image size the model resizes frames to.

We also see a big gap between class agnostic and class avare mAP, which shows that even when model correctly detects the object, it often struggles to classify it correclty. Class confusion is the highest for vans(38.3%), trucks(23.5%) and busses(21.7%). This is due to high class imbalance, as this is much rarer for dominant classes (people and cars).

We also see that mean best proposal IoU(0.784) is higher than mean final box IoU(0.753). This probably happens because we have both very small objects, such as people and big objects, for example buses, and the model struggles to find the optimal bounding box size to fit all of those.

The graphs and tables supporting those claims can be found in the Faster CNN colab
## Conclusion

> Summarize your findings here.
