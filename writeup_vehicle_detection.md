##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image11]: ./output_images/example_car.jpg
[image12]: ./output_images/example_not_car.jpg
[image21]: ./output_images/hog_car.jpg
[image22]: ./output_images/hog_car_vis.jpg
[image31]: ./output_images/window_sub_sampling.jpg
[image32]: ./output_images/multi_window_sub_sampling.jpg
[image4]: ./examples/sliding_window.jpg
[image51]: ./output_images/heatmap.jpg
[image52]: ./output_images/heatmap_detection.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_output.mp4


Histogram of Oriented Gradients (HOG)
-------------------------------------

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `train_calssifier.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image11]
![alt text][image12]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image21]
![alt text][image22]

HOG parameters
--------------

I focused on the HOG features as the the CarND course described them as the most important. Using color information is something I want to study more. Apart from identifying cars from background it could prove useful when identifying individual cars when we have overlapping car detections.

I got good results for LUV, HLS, YUV, YCrCb using all color channels. I tried using 8,9 and 12 directions and got good results for all (> 98 %). When lowering to 4 directions the accuracy shrunk to 97%. 

I started out with 8 for the pix_per_cell parameter and tried 4 and 16. Both alternatives yeilded lower accuracy (~97%). For the cell per block parameter I tried 2, 4 and 8. The value of 2 proved to be the optimal solution in combination with the other parameter choices. I suspect there is some interdependency.


In my pipeline I settled for 9 orientations and color space YCrCb and all color channels. I used 9 directions and 8 pixels per cell. The cell block parameter was 2.


Machine learning and training
-----------------------------

I opted for a linear SVM using only the HOG features. I did not explore any machine learning alternatives. This is something I want to futher investigate. The training is located in the file `train_classifier.py` in lines 190 to 193.

Sliding window search
---------------------

To identify cars in the image I used a sliding window search (The implementation can be found in `vehicle_detection.py` as function `find_boxes()` lines 42-104). I used multiple window sizes (1, 1.5, 2, 3, 4, 8). I scanned the lower part of the image as that's where cars can be expected to be found. The 1x scale window scan was very time consuming so I limited that to only serch for cars far away (y range= 400-500) 

Detections using single window size:

![alt text][image31]

Detections using multiple window sizes.

![alt text][image32]


Ultimately I searched on five window sizes using YCrCb 3-channel HOG features with the parameters described above. To speed up processing I parallelized the window search using ptyhon multiprocessing module.  

Here's a [link to my video result](./project_video_output.mp4)


False positives and detections
------------------------------

The sliding window search resulted in several overlapping detections. These detections where drawn to a heatmap. A threshhold in combination with the `scipy.ndimage.measurements.label()` identified bounding boxes which are mapped as car detections. 

To improve stability of detections and bounding boxes I used the four most recent bbox detections to construct the heatmap. This is reasonable as cars do not appear and disappear from frame to frame. This solution also filters away spurious detections of things that are not consistently identified as cars.  


Here's an example result showing the heatmap and the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Heatmap:
![alt text][image51]

Detection from heatmap:

![alt text][image52]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

