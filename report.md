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


## Conclusion

> Summarize your findings here.
