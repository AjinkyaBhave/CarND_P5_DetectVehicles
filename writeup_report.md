##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Training_Images.jpg
[image2]: ./output_images/HOG_Features.png
[image3]: ./output_images/SVM_HyperParam_Results.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

This document. The code is organised into two files:
1. classify_vehicles.py: implements feature extraction, classifier training and testing using project datasets.
2. detect_vehicles.py: implements sliding window on image, heatmaps, and vehicle tracking.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `classify_vehicles.py`(lines 12 to 29)

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Training Images][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I tested random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the final selection of all channels of the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![HOG Features][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters before finalising on the ones given above. I first started with the LAB colour space with 9 bins, 6 pixels per cell, 3 cells per block. This was one of the recommended configurations in the Dalal and Triggs (CVPR 2005) paper. The LAB space is an opponent colour space, which allows the HOG features to encode some colour information along with shape information. However, I found that the feature descriptor size was 15552, took a long time to train, and was slow when predicting a single image. 

So I experimented with balancing the feature size with the training accuracy, trying various combinations, till I froze the hyper-parameters given above. Te final feature vector size was 1188, which was a good balance between size, prediction accuracy, and computation speed. I found the YUV space to give a ~1.5% better classification performance compared to LAB. Augmenting the HOG features with colour histograms did not improve the classifier performance by an appreciable amount, compared to the feature vector size increase.Some of the hyper-parameter tuning results are shown in the figure below.

![Hyper-parameter Search][image3]

The parameters are defined in `classify_vehicles.py`(lines 108 to 114)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC()` with a `C` regularisation parameter of 0.01. The code is in `classify_vehicles.py` (lines 126 to 195). I tuned `C` using `GridSearchCV` (lines 178-179). I briefly experimented with the RBF kernel but finally settled on a linear SVM, given the speed of prediction and good accuracy of 99.4% on the supplied project dataset (GTI and KITTI). Using a smaller `C` value allows the classifier to mis-classify some training examples to allow better generalisation on the test set. When I set `C` to 1 or above, I was getting a lower test accuracy, as expected.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

