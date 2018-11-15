# Video Summarization

The code is designed for 2 scenarios: 
1. Using the VSUMM dataset
2. Streaming from webcam - takes live video stream for 2 minutes (change the duration as desired).  

Note: This is only the implementation of the proposed method in the paper [1]. 

Dependencies: 
Python and OpenCV
OS: Ubuntu 16.04 (14.04 works too)

To install opencv download anaconda and create a virtual environment. Then you can either follow: https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

or just do (much easier):

source activate virtualEnvironmentName

pip install opencv-python  ---->  in the terminal (inside the virtual environment). 


# Dataset
We make use of the VSUMM dataset which can be found in [2]. Download the videos and the user summaries of the 50 videos in the VSUMM dataset from the "Database" and "User Summary" links given in [2].
1. Create a folder called Videos in the current directory
2. Inside Videos, create 2 more folders called "dataset" and "UserSummary"
3. Place the 50 downloaded videos in the "database" folder. 
4. Place the user summaries in the "UserSummary" folder. 

# Feature Extraction methods

HOG feature extraction, VGG16, Resnet50 pretrained on ImageNet. 

# Running code
Would recommend running it in Anaconda spyder editor or anything analogous to it so that it is easier to debug. 


# References: 
[1] Anirudh, R., Masroor, A., & Turaga, P. (2016). Diversity Promoting Online Sampling for Streaming Video Summarization.

[2] https://sites.google.com/site/vsummsite/download
