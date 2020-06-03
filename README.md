**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1.1.1]: ./camera_cal/calibration1.jpg "Distorted"
[image1.1.2]: ./output_images/undistortImage.jpg "Undistorted"
[image2.1.1]: ./test_images/test3.jpg "Road Image"
[image2.1.2]: ./output_images/undistortlane.jpg "Road Undistorted"
[image2.2.1]: ./output_images/combinethreshImage.jpg "Binary Example"
[image2.3.1]: ./output_images/tuning_straight_warp_src.jpg "Warp Example"
[image2.4.1]: ./output_images/straight_lines1_detected.jpg "Fit Visual"
[image2.5.1]: ./output_images/curvature.jpg "Output Radius"
[image2.6.1]: ./output_images/lane_detect.jpg "Lane"
[image2.6.2]: ./output_images/output.jpg "Output Images"
[video1]: ./output_images/project_video_output.mp4 "Video"



---

#  README

## Camera Calibration

### 1. Calibration

The code for this step is contained in the first code cell of the IPython notebook located in `./Advanced_Lane_Detection.ipynb` from the `calibration()` function (located in lines 33 through 69 of the file called `helper.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the one of the calibration image using the `cv2.undistort()` function and obtained this result:

Distort:
![alt text][image1.1.1]

Undistort:
![alt text][image1.1.2]

## Pipeline (single images)

### 1. Distortion-corrected Image

After calibrating the camera, the first step in the pipeline is to undistort the image with `cv2.undistort()`:

Original Image:
![alt text][image2.1.1]

Undistorted 
![alt text][image2.1.2]

### 2. Filter Image

I used a combination of color and gradient thresholds to generate a binary image. The function which combined all filter is the `filter_image()` function inside `Advanced_Lane_Detection.ipynb`. The threshold can be adjust in that function. Each fitler function is located between lines 82 through 135 in `helper.py`).  

#### 2.1 Sobel Filter

Sobel Filter is used for edge detection which is done using `abs_sobel_thresh()` function . Here I used Sobel to find edges in both X and Y direction. Then I will combine them with other filter in a later process. The sobel filter is done using `cv2.Sobel()` function.

#### 2.2 Magnitude Theshold Filter

I have also use magnitude of the sobel as a threshold in the `mag_thresh()` function. This function use the magnitude of both the X and Y sobel. Therefore, it would detect both all the edge.

#### 2.3 Direction Theshold Filter

Another Sobel filter that was used is the `dir_theshold()` which find the edges within specific angle.

#### 2.4 Combined Filter

The last step for filtering process is to combined all filter together using some logical operator. It is included in the `filter_image()` function.

Here's an example of my output after going through all the filter:
![alt text][image2.2.1]

#### 3. Perspective Transform

Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 71 through 78 in the file `helper.py` (or, for example, `process_image()` function in IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  

In 2nd code cell of IPython notebook, I calibrated the src by using a straight line images by drawing the src point. The result of the the image is as follow:

![alt text][image2.3.1]

I chose the hardcode the destination points, The source and destination ponits are in the following manner:

```python
xsize, ysize = (straight.shape[1], straight.shape[0])
src = np.float32(
    [[xsize//2 - 50, 450 ],
    [xsize//2 + 53, 450],
    [xsize//2 + 465,ysize-10],
    [xsize//2 - 430,ysize-10]])
dst = np.float32(
    [[300, 0],
    [1000, 0],
    [1000, 720],
    [300, 720]])
```

It is worth mentioning that to draw the polygon in opencv, the data have to be in int32. After that, it was convert to float32 for `warper()` function.

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 590, 450  |   300, 0    |
| 693, 450  |  1000, 0    |
| 1105, 710 |  1000, 720  |
| 210, 710  |   300, 720  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

#### 4. Lane-Line Detection

After warping the image, I initially identify the lane pixel using `find_lane_pixels()` fucntion in `Advanced_Lane_Detection.ipynb`. This function will create a histrogram window, starting from the bottom of the image, and see which window has the highest number of white pixel. The one with maximum number of white pixels will be the starting lane line. After that, the window will move up and readjust the center of the window if necessary. It will record the coordinate of the white pixels into variable `leftx lefty rightx righty`.

Afterwards those coordinate will be used for polynomial fitting in the `fit_polynomial()` function. This will fit the datas using 2nd degree of polynomial through `np.polyfit()` function. Finally, I will get the coeffecient for both lines. To find the coefficient of the actual road distance, I multiple the datapoints with conversion ratio `xm_per_pixel` and `ym_per_pixel`. I assume that the lane width is 3.7 meters and that the road distance that is warped is 30 meters. Since the image height is 720 pixels and I warped the lane lines to be 700 pixels apart, the conversion value is as follow:

```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

After the first image of the video, it is not necessary to create a histrogram of window. I only needed to search for the point around the previous polynomial line. This is done with `search_around_poly()` and `fit_poly()` function in the ipynb file.

The result after searching is as shown below:

![alt text][image2.4.1]

#### 5. Radius of Curvature and Distance From Center

I did this in `measure_curvature()` function in `Advanced_Line_Detection.ipynb` file. To find the radius of curvature, all I need is the polynomial coefficient of each line. After that, I average the two curvature.

To find the distance from center, I average of the pixel position of the two lines. Assuming that the camera is mount exactly at the center of the car, the distance of center would be the difference between that average and the pixel center of the image. After that, I multiply by ratio xm_per_pixel to convert it to real distance.

The curvature is shown in the image as shown below:

![alt text][image2.5.1]

#### 6. Projection of Road and Result

I implemented the lane projection with `project_lane()` function in my `Advanced_Lane_Detection.ipynb` file which will draw the detected lane onto original image. Here is an example of my result on a test image:

![alt text][image2.6.1]

The entire pipeline is done inside the `process_image()` function. This function will also determine if the curvature is too high or if the algorithm lose track of the lane. If it does, then it will go back to searching for the lane line with histrogram search and not just look around previous lane line polynomial.

The overall result is as shown:
![alt text][image2.6.2]

---

### Pipeline (video)

#### 1. Video Output

Here's the output video:
![link to my video result][video1]

---

### Discussion

#### 1. Problem

I face a few challenges along the way which I will list them here.

First, the image filtering process will not work well for every frames of the images. Certain threshold work well when there is tree shadows while it might not work well when there is high sunlight , and vice-versa.

Second, the structure of the code may not be how I want it to be. Intially, I plan to be make those function more independent by not relying on having class variable inside the function, but instead take them as input and give output. However, because I haven't plan those function well enough and for the ease of debugging, those function access the class variable directly. Therefore, if those variable have not been created, it would give error. 

#### 2. Futher Recommendation

In the future, I would try to tune the threshold better with different images and lighting condition to find the best threshold. I could add more logic to give different weight on each filter type depending on the environment.

Secondly, the function should be put inside another file to keep the code cleaner. The function which involve class object could be class methods to keep it organize.