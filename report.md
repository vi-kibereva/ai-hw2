---
title: Homework 2 -- Multiple Object Video Detection
authors: Olena Tumak, Viktoriya Kibyeryeva
course: Artificial Intelligence · UCU
date: 2026-11-16
---

### EDA:

We've started our work with EDA to get to know our data and prepare it for our models.
We've found:
1. Significant class imbalance: cars(44.0%) and people(31.1%) together take more than 75% of the dataset, while buses are only 0.8 %. This may cause models to overfit to predicting only dominant classes and discard all other ones. For now we decided to not use any data augmentation to see how the models will perform without them, but we will include those later if we see the need for those.
2. Below you can see sizes distribution across different object types and clips. We can see that those wary highly, but the biggest part of object is at around 32 to 96 px.
![Sizes distribution](images/sizes_distribution.png)
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

Box positive fraction of 0.5 raised the result but only slightly (from initial 0.120 to 0.122) but when we lowered it to 0.10 mAP dropped to 0.106 because second stage trained mostly on background and wasn't able to classify objects accurately.

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

We also see a big gap between class agnostic and class average mAP, which shows that even when model correctly detects the object, it often struggles to classify it correctly. Class confusion is the highest for vans(38.3%), trucks(23.5%) and buses(21.7%). This is due to high class imbalance, as this is much rarer for dominant classes (people and cars).

We also see that mean best proposal IoU(0.784) is higher than mean final box IoU(0.753). This probably happens because we have both very small objects, such as people and big objects, for example buses, and the model struggles to find the optimal bounding box size to fit all of those.

The graphs and tables supporting those claims can be found in the [Faster CNN colab](./notebooks/FasterCNN.ipynb)

### One-stage anchor based (RetinaNet)

For our one-stage anchor based model we chose RetinaNet because on the contrary to other one-stage detectors, which struggle with small object it integrates Feature Piramid Network. And this is very important to us as we have objects of very different sizes.

We used pre-trained weights from RetinaNet_ResNet50_FPN_Weights.COCO_V1 for the same reasons as for FasterCNN: we don't have enough data and time to justify training the model from scratch, and those weights suit us best as they are pre-trained on similar objects to ours.

First step of tuning this model was checking and comparing anchor sizes. At image size 1536 the median size of our object is 38 px and 5% is 16.9 px. Because of this we can't use torchvision default anchors, as they start at 32px. So we will use our custom anchor sizes, which start at 5% and grow twice at each level. This raised our coverage from default 86% to 97.8%.

The next step of tuning the parameters was checking if using custom anchors yielded any results. Sadly, this only weakened model's performance. The main reason is that even though our anchors covered 97.6% of all object they were too small for medium and large objects and caused AP_medium and AP_large to drop, which dropped mAP too.

Our next step was tuning foreground and background IoU. Because our dataset contains frames with a big amount of objects which can overlap, we wanted to check if relaxing those thresholds a bit will make it easier for model to capture those. Lowering foreground IoU from default 0.5 to 0.4 and background IoU from 0.4 to 0.3 did increase mAP but only by 0.03(from 0.63 on initial model to 0.66). We tried lowering both of those 0.1 more but that only lowered the score.

The other parameter we decided to tune focal loss focusing parameter because our dataset is highly imbalanced and this focal loss controls how much the model is penalized for missing rare classes in comparison to popular classes. Setting focal_gamma to 1.0 really improved our results. 

### Comparison of two-stage and one-stage models:

Removing the region proposal stage can cause high foreground-background class imbalance. In two-stage models first stage discards background boxes so second stage can focus solely on classifying and adjusting those boxes. But when we remove the first stage the other one is forced to calculate anchors across the whole image and can easily become biased to classify everything as background. Focal loss is the main mechanism to fix this.

![Model comparison](./images/comparison.png)

Overall Performance and Efficiency Metrics
| Model | mAP | AP50 | AP75 | AP_small | AP_medium | AP_large | AR500 | ms_per_img | fps | train_min |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RetinaNet initial** | 0.160 | 0.322 | 0.139 | 0.037 | 0.145 | 0.294 | 0.285 | 97.803 | 10.225 | 54.704 |
| **RetinaNet custom anchors** | 0.142 | 0.327 | 0.107 | 0.051 | 0.130 | 0.249 | 0.294 | 95.554 | 10.465 | 54.832 |
| **Faster R-CNN tuned** | 0.201 | 0.432 | 0.161 | 0.085 | 0.189 | 0.309 | 0.367 | 98.633 | 10.139 | 77.099 |

Per-Class Average Precision (AP) Breakdown
| Model | AP_person | AP_bicycle | AP_car | AP_van | AP_truck | AP_tricycle | AP_bus | AP_motor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RetinaNet initial** | 0.178 | 0.053 | 0.355 | 0.102 | 0.098 | 0.075 | 0.322 | 0.096 |
| **RetinaNet custom anchors** | 0.177 | 0.063 | 0.329 | 0.075 | 0.071 | 0.084 | 0.241 | 0.096 |
| **Faster R-CNN tuned** | 0.215 | 0.091 | 0.387 | 0.133 | 0.145 | 0.131 | 0.360 | 0.147 |

Out of the two anchor-based models, FasterCNN showed the best performance with mAP = 0.201 and AP50 = 0.43.
It also has AP small more than twice greater than initial anchor RetinaNet's one. 
AP small is the main advantage of RetinaNet with custom anchors but due to them it lost too much on medium and large objects and its mAP turned out to be smaller than initial one.

From per-class breakdown we see that FasterCNN consistently outperformed the model across all classes and achieved the best results on cars, buses and people. This was expected for cars and people because those two are majority classes but was unexpected for buses which are a minority class. But buses are also a class with the greatest average size, so that probably made it easier for the model to learn them.

### Further experiments

To get the performance better than separate scores of two anchor based model we decided to fuse them together. Even though errors of those models are correlated as there is a similarity in how they work, we thought that together they can get rid of some of them.

Our candidates for fusion were FasterCNN with image sizes 1280 and 1960, those two with flipped data and RetinaNet with both initial and tuned anchors.
To combine predictions from different models we used Weighted Box Fusion. This way we group all overlapping detections and check if IoU is greater than our threshold. Then we calculated final bounding box position based on models' confidence. To account for the difference in models' performance we assigned each model a weight, so that worse models don't pull the score down.

To check which models will increase the score we used greedy forward selection. We took our best performing FasterCNN variations as baseline and for each further model checked if adding them will improve our score. As a result our final ensemble consists of FasterCNN with image size 1280, FasterCNN with image size 1280 and flipped inputs, FasterCNN with image size 1920 and flipped input and RetinaNet with initial anchors.This way we achieved a score 0.2205 on public leaderboard.

However, it still failed to perform better that Query based model.

## Conclusion

> Summarize your findings here.
