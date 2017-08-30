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
[image2]: ./output_images/HOG_Features.jpg
[image3]: ./output_images/SVM_HyperParam_Results.jpg
[image4]: ./output_images/Sliding_Windows_2.jpg
[image5]: ./output_images/Sliding_Windows_1.5.jpg
[image6]: ./output_images/Sliding_Windows_1.jpg
[image7]: ./output_images/Vehicle_Detection_Pipeline.jpg
[image8]: ./output_images/False_Positive_Window.jpg
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

I trained a linear SVM using `LinearSVC()` with a `C` regularisation parameter of 0.01 and test set of 20% of the total examples. The code is in `classify_vehicles.py` (lines 126 to 195). I tuned `C` using `GridSearchCV` (lines 178-179). I briefly experimented with the RBF kernel but finally settled on a linear SVM, given the speed of prediction and good final accuracy of 99.4% on the supplied project dataset (GTI and KITTI). Using a smaller `C` value allows the classifier to mis-classify some training examples to allow better generalisation on the test set. When I set `C` to 1 or above, I was getting a lower test accuracy, as expected. I have also saved the classifier and the scaler models to disk at the end of the training (lines 188-189), so I can re-use them in the vehicle detection phase.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code to slide windows and subsample the image for HOG features is modified from `find_cars()` given in the lessons. It is implemented in `detect_vehicles.py` (lines 76 to 183). The changes are related to optimising the code to use only HOG features, adding visualisation for debugging, and checking for the classifier confidence before accepting a detection as valid (line 133).

I experimented with scales between 3 to 1 and finally settled on [2, 1.5, 1] as the three scales for the sliding windows. I applied the scales from the lower to the upper parts of the image, since I do not expect large cars far away from the camera. I chose the overlap after manually looking at the video over multiple iterations to see which combination gives the most reliable detections at each scale. I also chose different overlaps in the x and y directions for each scale: [75%, 75%, 50%]. This helps to minimise the computation time while detecting images, since the image size to search over is reduced. The final parameters for the sliding windows are defined in lines 217 to 231. The search regions for each of the scales are shown in the images below.

![Sliding Windows: Scale 2][image4]

![Sliding Windows: Scale 1.5][image5]

![Sliding Windows: Scale 1][image6]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features, which provided the required accuracy. I chose the image search region and overlap for each of the scales in a way that reduces the search space. This is more efficient than naively searching the entire bottom image for all scales (large car regions appear near the bottom while far-away/smaller car regions appear near the middle of the image). I also tuned the window overlap for each scale after running the pipeline over multiple test images. Here are some example images which show how the bounding boxes and heatmap evolve over multiple frames:

![Vehicle Detection Pipeline][image7]

The first row shows the bounding boxes detected. However, there is no vehicle detected because the heatmap threshold is not reached yet, since it is calculated over multiple frames. So there is no detection in the first frame. The second row shows the output after three frames. The two vehicles are now detected and the heatmap correspondingly shows the region of the accumulated detections over previous frames. The third row shows the final frame in the series. where the heatmap is now maximised, and the two vehicles are strongly detected. Notice that the bounding box of the white car is now more accurate compared to the previous row because of added information over the previous time steps.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To combine the overlapping bounding boxes obtained from detection at multiple scales, I used the *heatmap* approach outlined in the lectures. I recorded the window coordinates of positive detections in each frame of the video.  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions. The threshold is tuned assuming aggregation over multiple frames. I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, as suggested in the lesson.  I  assumed each blob corresponded to a unique vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code is in `track_vehicles()` (lines 44 to 49) and the functions referenced there. Since they are identical to the code snippets given in the lecture, I will not describe them in more detail here. 
 
My approach to filtering false positives is based on a number of optimisations. The first is to check the confidence of the classifier prediction and reject it if it is below a threshold. This is implemented in `find_vehicles()` (line 140). The value of `svc_conf_thresh` was obtained after analysing the return value of `svc.decision_function` over a large number of detections at all scales over multiple images in the video. The decision function calculates the distance of the test feature from the SVM margin. The more confident the classifier is, the farther the test point is from the margin, and the larger is the return value. Since the value was close to 1 in most cases of good detection, I used this as the value to check against (line 243) and to balance rejecting weaker detections vs. allowing too many false positives.

The second is to aggregate the window detections over a series of images, rather than on a single image. This eliminates any transient detections over multiple frames. By empirically defining a threshold for the heat map (line 244), I was able to eliminate the false positives that popped up earlier in the pipeline. 

The third is to tightly define the region of image to be searched at the three window scales. I found that there were many false positives at the bottom of the image at scale 1 . By enforcing the search region to a narrow band near the middle of the image, the window search detected valid cars and drastically reduced the false positives at that scale. Similar region tuning was applied for the 1.5 and 2 window scales.

The fourth is to manually create negative training examples from the false positives and retrain the classifier with the new examples included. This helped to reduce the commonly occuring false positives (yellow road markings and green traffic signs), and made the classifier more robust.

An example false positive (left of black car), which was eliminated with this combined approach, is shown below:

![False Detection][image8]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most time was spent in deciding the thresholds for the heat map and the prediction confidence, as well as the window scales, search regions, and overlap. These were critical in reducing the false positives to an acceptable number in the final implementation. The current pipeline still has a very small number of false positives (the single one in the video appears at ~48 secs just when the white car goes out of frame). Deciding the size of the queue over which to store the bounding boxes also took time to tune empirically.

I am not happy with the accuracy of the bounding boxes in the current implementation. Time permitting, I would have liked to get a tighter overlapping region for each detected vehicle. I might do this later using the non-maximum suppression method for overlapping boxes. My pipeline is not robust to occluding vehicles and clubs the two bounding boxes together when one car overtakes the other. Currently there is a very transient false positive at the end of the video but I cannot guarantee that more will not occur on a different video or under different lighting conditions. Training the classifier on a more diverse dataset, possibly Udacity's, would mitigate this problem.

The code does not run in real-time. To make it so, I would first investigate the OpenCV HOG detector, which is supposed to be an order of magnitude faster. I would also compare a deep learning approach like YOLO(2) which is much faster on GPU-based implementations.

