
## Project Info

### Title
Lane Detection

### Department 
Electrical Engineering


## Introduction

### Overview

Our project is based on the idea of Machine Learning. Machine Leaning is a type of artificial intelligence that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values. 

So why Autonomous Car? -The idea is to make an autonomous car which can sense signals such as red and green, it can also sense obstacles such as cars, it can the route supplied to it (it can follow a straight road or turn having boundary). 

### Purpose

It is well known fact that error is an inevitable part of an animal’s life. The error can be caused by both external and internal factors. Sometimes due to these errors it may cause an accident. To reduce these human errors, we have to make autonomous technology our future. 

We believe that our project if implemented practically can greatly reduce the number of road accidents and reduce the number of signal jumpers! 

- Lane detection systems will **quickly alert the driver** if their vehicle should cross over the line dividing lanes, thereby helping to avoid an accident. Like many other safety features, creators of this technology claim that it will help prevent car accidents. 

## Literature Survey

### Existing Problem
- Conventional cars being non-autonomous are more prone to road accidents subject to human error. Autonomous cars on the other end have an edge over the conventional (non-autonomous cars). 
- Large number of conventional cars resulting in greater risk of road accidents and present-day cars also cause huge amount of      pollution. 
- Huge cost of manufacturing autonomous cars. Ultimately adding to the cost of the car. 
### Proposed Solution
- Autonomous Driving Car is one of the most disruptive innovations in AI. Fueled by Deep Learning algorithms, they are continuously driving our society forward and creating new opportunities in the mobility sector. An autonomous car can go anywhere a traditional car can go and does everything that an experienced human driver does. But it’s very essential to train it properly. One of the many steps involved during the training of an autonomous driving car is lane detection, which is the preliminary step. So, we are going to learn how to perform lane detection using videos, the source code and result of this program is reflected in our project. 
## Theoritical Analysis & Requirements:

### Software Design

- **Python Programming Language** <br>

    Python is as software goes, a fairly easy programming language to learn and judging by the size of the open-source community supporting it, it is extremely popular. It’s popularity with autonomous vehicle engineers is largely due to the various available resource libraries such as AI, deep learning and data visualization. 

    ![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.004.png)

    [https://robots.net/tech/30-must-have-online-sources-to-master-python-programming-easily/ ](https://robots.net/tech/30-must-have-online-sources-to-master-python-programming-easily/)

    <br>

- **OpenCV** <br>
    **OpenCV** is an open source, cross platform and free to use open-source library (under BSD license) for computer vision, machine learning, and image processing. It is a highly optimized library with focus on real-time applications. It can support Python, .Java, C++ etc. It was originally developed by Intel. It is one of the most widely used library for motion detection(tracking), video 

    recognition, image recognition and deep learning (for facial recognition). It can process images and videos to identify objects, faces, or even the handwriting of a human 

    ![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.005.png)

    [https://www.analyticsvidhya.com/blog/2021/05/computer-vision-using-opencv-with-practical-examples/ ](https://www.analyticsvidhya.com/blog/2021/05/computer-vision-using-opencv-with-practical-examples/) <br><br>

- **NumPy** <br><br>

    NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. Moreover, NumPy forms the foundation of the Machine Learning stack. 

    ![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.006.png)

    [ https://towardsdatascience.com/how-to-create-numpy-arrays-from-scratch-3e0341f9ffea ](https://towardsdatascience.com/how-to-create-numpy-arrays-from-scratch-3e0341f9ffea) <br><br>

- **PyCharm IDE** <br><br>

    PyCharm is a dedicated Python Integrated Development Environment (IDE) providing a wide range of essential tools for Python developers, tightly integrated to create a convenient environment for **productive Python, web, and data science development**. 

    PyCharm being accepted widely among big companies for the purpose of Machine Learning is due **to its ability to provide support for important libraries** like OpenCV, NumPy. 

    ![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.007.png)

    [https://towardsdatascience.com/4-tips-to-get-the-best-out-of-pycharm-99dd5d01932d ](https://towardsdatascience.com/4-tips-to-get-the-best-out-of-pycharm-99dd5d01932d)

## Block Diagram

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.008.png)

- **Capturing and decoding video file:** We will capture the video using Video Capture object and after the capturing has been initialized every video frame is decoded (i.e., converting into a sequence of images). 
- **Grayscale conversion of image:** The video frames are in RGB format, RGB is converted to grayscale because processing a single channel image is faster than processing a three-channel colored image. The images should be converted into gray scaled ones in order to detect shapes (edges) in the images. This is because the Canny edge detection measures the magnitude of pixel intensity changes or gradients.  

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.009.png)

- **Gaussian Smoothing (Gaussian Blur):** When there is an edge (i.e., a line), the pixel intensity changes rapidly (i.e., from 0 to 255) which we want to detect. But before doing so, we should make the edges smoother. As you can see, the above images have many rough edges which causes many noisy edges to be detected. Noise can create false edges, therefore before going further, it’s imperative to perform image smoothening. Gaussian filter is used to perform this process. The Gaussian Blur takes a kernel size parameter (they must be positive and odd) The bigger the kernel size value is, the blurrier the image becomes. 
- Bigger kernel size value requires more time to process. We should prefer smaller values if the effect is similar. 

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.010.png)

- **Canny Edge Detector:** It computes gradient in all directions of our blurred image and traces the edges with large changes in intensity. 

cv2.Canny takes two threshold values it is essential to filter out the edge pixel with the weak gradient value and preserve the edge with the high gradient value. Thus, two threshold values are set to clarify the different types of edge pixels, one is called high threshold value and the other is called the low threshold value. If the edge pixel’s gradient value is higher than the high threshold value, they are marked as strong edge pixels. If the edge pixel’s gradient value is smaller than the high threshold value and larger than the low threshold value, they are marked as weak edge pixels. If the pixel value is smaller than the low threshold value, they will be suppressed. 

The double thresholds are used as follows: 

- If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge 
- If a pixel gradient value is below the lower threshold, then it is rejected. 
- If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected to a pixel that is above the upper threshold. 

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.011.png)

- **Region of Interest:** This step is to take into account only the region covered by the road lane. A mask is created here, which is of the same dimension as our road image. Furthermore, bitwise AND operation is performed between each pixel of our canny image and this mask. It ultimately masks the canny image and shows the region of interest traced by the polygonal contour of the mask. 

When finding lane lines, we don't need to check the sky and the hills. 

Roughly speaking, we are interested in the area surrounded by the red lines below: 

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.012.png)

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.013.jpeg)

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.014.jpeg)

- **Hough Transform Line Detection:** The Hough Line Transform is a transform used to detect straight lines. The Probabilistic Hough Line Transform is used here, which gives output as the extremes of the detected lines. 

cv2.HoughLinesP is used to detect lines in the edge images. There are several parameters that need to be checked. 

cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap). 

rho: Distance resolution of the accumulator in pixels. theta: Angle resolution of the accumulator in radians. 

threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes (> threshold). 

minLineLength: Minimum line length. Line segments shorter than that are rejected. 

maxLineGap: Maximum allowed gap between points on the same line to link them. 

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.015.png)

## **Implementation process of Source Code:** 

**Libraries required for this task:** 

- **NumPy: It is installed along with OpenCV package in pycharm IDE.** 
- **OpenCV: It can be installed in two ways, using anaconda or using pip.** 

To install using anaconda, type: ```conda install -c conda-forge opencv```
or to install using pip, type: ```pip install opencv-python``` into our command line/terminal.

- Import the required libraries import cv2 

import numpy as np 

- **The canny function calculates derivative in both x and y directions, and according to that, we can see the changes in intensity value. Larger derivatives equal to High intensity(sharp changes), Smaller derivatives equal to Low intensity(shallow changes):** 

def canny(*img*): 

- Convert the image color to grayscale 

```gray = cv2.cvtColor(img, cv2.COLOR\_RGB2GRAY)```

- Reduce noise from the image 

```kernel=3```
```blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)``` 
```canny = cv2.Canny(blur, 40, 120) return canny ```

- **Masking our canny image after finding the region of interest:** 

```
def region\_of\_interest(*canny*): 

height = canny.shape[0] 

quad = np.array([ 

[(200, height),(200,450),(1100,450), (1100, height)] ], np.int32) 

mask = np.zeros\_like(canny)
```

- Fill poly-function deals with multiple polygon cv2.fillPoly(mask, quad, 255) 
- Bitwise operation between canny image and mask image masked\_image = cv2.bitwise\_and(canny, mask) 

```return masked\_image```

- **We are going to find the coordinates of our road lane:** 

```
def make\_points(*image*, *line*): 

slope, intercept = line 

y1 = image.shape[0] 

y2 = *int*(y1 \* (3 / 5)) 

x1 = *int*((y1 - intercept) / slope) x2 = *int*((y2 - intercept) / slope) return np.array([x1, y1, x2, y2]) 
```

- **Differentiating left and right road lanes with the help of positive and negative slopes respectively and appending them into the lists, if the slope is negative then the road lane belongs to the left-hand side of the vehicle, and if the slope is positive then the road lane belongs to the right-hand side of the vehicle:** 

```
def average\_slope\_intercept(image, lines): 

left\_fit = [] 

right\_fit = [] 

if lines is None: 

return none 

for line in lines: 

for x1, y1, x2, y2 in line: 

fit = np.polyfit((x1, x2), (y1, y2), 1) 

slope = fit[0] 

intercept = fit[1] 

if slope < 0: # y is reversed in image left\_fit.append((slope, intercept)) 

else: 

right\_fit.append((slope, intercept)) 

if len(left\_fit) and len(right\_fit): 

left\_fit\_average = np.average(left\_fit, axis = 0) right\_fit\_average = np.average(right\_fit, axis = 0) left\_line = create\_coordinates(image, left\_fit\_average) right\_line = create\_coordinates(image, right\_fit\_average) return np.array([left\_line, right\_line]) 
```

- **Fitting the coordinates into our actual image and then returning the image with the detected line (road with the detected lane)** 

```
def display\_lines(*img*, *lines*): 

line\_image = np.zeros\_like(img) 

if lines is not None: 

for x1, y1, x2, y2 in lines: 

cv2.line(line\_image, (x1, y1), (x2, y2), (0, 255, 255), 10) return line\_image 
```

- **Firstly, the video file is read and decoded into frames and using Houghline method the straight line which is going through the image is detected. Then we call all the functions.**

### test on image 
```
test\_img = cv2.imread("test\_image.jpg") 

while test\_img is not None: 

test\_img = cv2.resize(test\_img, (1200, 700)) 

canny\_image = canny(test\_img) 

cropped\_canny = region\_of\_interest(canny\_image) 

lines = cv2.HoughLinesP(cropped\_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

averaged\_lines = average\_slope\_intercept(test\_img, lines) 

line\_image = display\_lines(test\_img, averaged\_lines) 

combo\_image = cv2.addWeighted(test\_img, 0.8, line\_image, 1, 1) cv2.imshow("Output Image", combo\_image) 

- When the below two will be true and will press the 'q' on 
- our keyboard, we will break out from the loop 

if cv2.waitKey(10) & 0xFF == ord('q'): 

break 

#test on video 

cap = cv2.VideoCapture("test2.mp4") 

while(cap.isOpened()): 

ret, frame = cap.read() 

if ret == True: 

canny\_image = canny(frame) 

cropped\_canny = region\_of\_interest(canny\_image) 

lines = cv2.HoughLinesP(cropped\_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

averaged\_lines = average\_slope\_intercept(frame, lines) 

line\_image = display\_lines(frame, averaged\_lines) 

combo\_image = cv2.addWeighted(frame, 0.8, line\_image, 1, 1) cv2.imshow("Output Video", combo\_image) 

- 10ms will wait for the specified time only between each frames 

if cv2.waitKey(10) & 0xFF == ord('q'): 

break 

else: 

break 
```

## Source Code: 

```
import cv2 

import numpy as np 

def make\_points(image, line): 

slope, intercept = line 

y1 = *int*(image.shape[0])  # bottom of the image 

y2 = *int*(y1 \* 3 / 5) # slightly lower than the middle x1 = *int*((y1 - intercept) / slope) 

x2 = *int*((y2 - intercept) / slope) 

return [[x1, y1, x2, y2]] 

def average\_slope\_intercept(image, lines): 

left\_fit = [] 

right\_fit =  [] 

if lines is None: 

return None 

for line in lines: 

for x1, y1, x2, y2 in line: 

fit = np.polyfit((x1, x2), (y1, y2), 1) slope = fit[0] 

intercept = fit[1] 

if slope < 0: # y is reversed in image 

left\_fit.append((slope, intercept)) else: 

right\_fit.append((slope, intercept)) 

- add more weight to longer lines if len(left\_fit) and len(right\_fit): 

left\_fit\_average = np.average(left\_fit, axis=0) right\_fit\_average = np.average(right\_fit, axis=0) left\_line = make\_points(image, left\_fit\_average) right\_line = make\_points(image, right\_fit\_average) averaged\_lines = [left\_line, right\_line] 

return averaged\_lines 

def canny(img): 

gray = cv2.cvtColor(img, cv2.COLOR\_RGB2GRAY) kernel = 3 

blur = cv2.GaussianBlur(gray, (kernel, kernel), 0) canny = cv2.Canny(blur, 40, 120) 

return canny 

def display\_lines(img, lines): 

line\_image = np.zeros\_like(img) 

if lines is not None: 

for line in lines: 

for x1, y1, x2, y2 in line: 

cv2.line(line\_image, (x1, y1), (x2, y2), (0, 255, 255), 10) #Draws a line #image = cv2.line(image, start\_point, end\_point, color, thickness in px) #(0,255,255) is BGR code for yellow![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.018.png) 

return line\_image 

def region\_of\_interest(*canny*): 

height = canny.shape[0] 

quad = np.array([ 

[(200, height),(200,450),(1100,450), (1100, height)] ], np.int32) 

mask = np.zeros\_like(canny) 

- Fill poly-function deals with multiple polygon cv2.fillPoly(mask, quad, 255) 
- Bitwise operation between canny image and mask image masked\_image = cv2.bitwise\_and(canny, mask) 

return masked\_image 

#test on image 

test\_img = cv2.imread("test\_image.jpg") 

while test\_img is not None: 

test\_img = cv2.resize(test\_img, (1200, 700)) 

canny\_image = canny(test\_img) 

cropped\_canny = region\_of\_interest(canny\_image) 

lines = cv2.HoughLinesP(cropped\_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

averaged\_lines = average\_slope\_intercept(test\_img, lines) 

line\_image = display\_lines(test\_img, averaged\_lines) 

combo\_image = cv2.addWeighted(test\_img, 0.8, line\_image, 1, 1) cv2.imshow("Output Image", combo\_image) 

if cv2.waitKey(10) & 0xFF == ord('q'): 

break 

#test on video 

cap = cv2.VideoCapture("test2.mp4") 

while(cap.isOpened()): 

ret, frame = cap.read() 

if ret == True: 

canny\_image = canny(frame) 

cropped\_canny = region\_of\_interest(canny\_image) 

lines = cv2.HoughLinesP(cropped\_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) 

averaged\_lines = average\_slope\_intercept(frame, lines) 

line\_image = display\_lines(frame, averaged\_lines) 

combo\_image = cv2.addWeighted(frame, 0.8, line\_image, 1, 1) cv2.imshow("Output Video", combo\_image) 

if cv2.waitKey(10) & 0xFF == ord('q'): 

break 

else: 

break 
```

### **Project Result Pictures:** 
![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.019.jpeg)

![](https://github.com/RonojitPal/LaneDetection/blob/main/Images/Aspose.Words.106d7b49-b4bb-4c84-8cd2-6c03cced8997.020.jpeg)

## Advantages

1. **Increased safety** 

**Automated reactions, fewer accidents. According to the National Highway Traffic Safety Administration (NHTSA),[ 94 percent of accidents a](https://www.eia.gov/analysis/studies/transportation/automated/pdf/automated_vehicles.pdf)re related to human error. Automated vehicles can remove human error and prevent a huge majority of accidents.** 

**To put this metric into perspective, on average,[ 222 trucks are involved in an accident every 100 million vehicle miles.](https://www.eia.gov/analysis/studies/transportation/automated/pdf/automated_vehicles.pdf) With automated driving, the truck accident rate could fall to** eight crashes per 100 million miles **by 2040. Liability coverage could also[ drop by 10 percent.](https://www.forbes.com/sites/jeffmcmahon/2016/02/19/autonomous-vehicles-could-drive-car-insurance-companies-out-of-business/2/)** 

**With AVs, humans’ slower reaction time, distracted driving, etc. can be eliminated from the equation, drastically reducing accidents and increasing safety.** 

2. **Greater efficiency *Reduced congestion*** 

**Automated driving leads to fewer accidents. With fewer traffic incidents comes fewer reasons to slow traffic. This translates to fewer instances of police cars and wrecked vehicles on the side of the road, blocked lanes, and so on. As a result, road congestion is reduced by about[ 25 percent.](https://www.eia.gov/analysis/studies/transportation/automated/pdf/automated_vehicles.pdf) No more[ rubbernecking!](https://en.wikipedia.org/wiki/Rubbernecking)** 

**Truck platooning can also reduce highway congestion. In case you aren’t familiar with the term,[ platooning i](https://en.wikipedia.org/wiki/Platoon_\(automobile\))s when autonomous trucks and cars are grouped together using electrical coupling, enabling simultaneous braking and acceleration. This avoids the “accordion” effect of vehicles speeding up or slowing down at unpredictable times. Because high-tech sensors can react 90 percent faster than humans, platooning allows the distance between trucks to fall from 50 meters to[ 15 meters.](https://www.2025ad.com/in-the-news/blog/truck-platooning-infographic/#%26gid%3D1%26pid%3D2)** 

***Increased lane capacity*** 

**With the ability to operate at higher speeds and reduced space between vehicles, platooning can also lead to[ 500 percent greater lane capacity,](http://www.rand.org/content/dam/rand/pubs/research_reports/RR400/RR443-2/RAND_RR443-2.pdf) vehicles per lane per hour.** 

**Lane capacity can also increase with the adoption of adaptive cruise control. Adaptive cruise control automatically adjusts vehicle speed to maintain a safe distance between cars and trucks ahead. This allows drivers to stay on cruise control for miles and avoid having to speed up or slow down depending on traffic. When a majority of vehicles adopt this technology,[ lane capacity can increase by 80 percent.](https://www.eia.gov/analysis/studies/transportation/automated/pdf/automated_vehicles.pdf)** 

***Real-time route optimization*** 

**Follow the best routes according to real-time information. Autonomous vehicles may be able to read the condition of the roads in real-time and redirect the routes accordingly.** 

**Vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) are developing technologies that enable vehicles to exchange safety and mobility information with one another and surrounding infrastructure. Vehicles can receive information like road conditions while en-route and shift accordingly or autonomously coordinate behavior with other vehicles and infrastructure like at intersections.** 

**Using V2V and V2I to determine optimal routes can reduce the number of miles driven, saving time and fuel use.** 

3. **Less energy consumption**

Greater efficiencies lead to more energy savings for your fleet. Fuel consumption of commercial light trucks, buses and heavy-duty freight trucks is expected to reduce up to 18 percent by 2050. 

As the risk of accidents decreases, so does the necessary weight of vehicles. Vehicles could become[ 25 percent lighter by 2030 w](http://www.rand.org/content/dam/rand/pubs/research_reports/RR400/RR443-2/RAND_RR443-2.pdf)hile maintaining their size. This could save up to **7 percent in fuel consumption**. 

In addition, eco-driving technology, which automates and optimizes driving like cruise control and smooth acceleration and deceleration, can[ improve fuel economy up to 10 percent.](http://www.rand.org/content/dam/rand/pubs/research_reports/RR400/RR443-2/RAND_RR443-2.pdf) 

Of course, despite the potential for the reduction in energy consumption, there’s also the possibility it could increase. This could occur due to reduced travel cost, faster highway speeds, longer commutes and increased accessibility of driving like to the elderly or disabled. According to an[ NREL study,](https://link.springer.com/chapter/10.1007%2F978-3-319-05990-7_13) depending on how these factors play out, **fuel consumption could decrease by 80 percent or increase up to 200 percent by 2050**. 

Although it is difficult to predict the effect of automation on fuel consumption overall, automation is expected to increase fuel savings for trucks. 

4. **More productivity**

The power of multi-tasking. There are varying levels of automated vehicles, but more advanced levels only require the driver to monitor the drive to make sure it goes smoothly. As a result, drivers in autonomous vehicles may not have to give their full, constant attention to driving vehicles and can shift their focus to other fleet related tasks like paperwork. Your fleet accomplishes more in a shorter period of time.

**In addition, your fleet vehicles may be able to travel for longer time periods now that a driver isn’t controlling the vehicle and subject to drowsiness. Although there are yet to be policies or regulations surrounding this issue, it’s possible there will be extended allowable driving time for each driver.** 

## Conclusion and Future Work

As of now we have successfully found out the lane detection from videos, in future we will come up with the latest strong algorithms such that with so much of precision we can smoothly detect every other parameter from its surroundings. There are lots of other factors if we compare our autonomous car in real life situation i.e., emergency lane changes, early obstacle avoidance, temporary stops, pile up avoidance, weather condition etc. currently we are working over other parameters not but the least we are highly grateful to our team as within a short span of time we have done something really wonderful which not only boost us but also gives us a lot of thirsty knowledge and enjoyment. Thankful to all those open-source platforms and various websites from where we have quenched out the idea and knowledges.

Again, thanks to everyone for giving us a chance to dive for new adventures.

## Bibliography 

**1.[ www.geeksforgeeks.org ](http://www.geeksforgeeks.org/)2[.www.google.com ](http://www.google.com/)3[.www.github.com ](http://www.github.com/)4[.www.python.org ](http://www.python.org/)5[.www.jetbrains.com ](http://www.jetbrains.com/)6[.www.youtube.com ](http://www.youtube.com/)7[.www.stackoverflow.com ](http://www.stackoverflow.com/)8[.www.coderedirect.com ](http://www.coderedirect.com/)9[.www.codegrepper.com** ](http://www.codegrepper.com/)**

