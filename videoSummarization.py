#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:52:25 2018

@author: kaushik
"""

import numpy as np
import cv2
import os, shutil
import time
import glob
from pywt import wavedec2
import time

########################################
### EXTRACT FEATURE
########################################

def featureExtraction(Funcframe):
#    win_size = (64, 128)
#    Funcframe = cv2.resize(Funcframe, win_size)
#    Funcframe = np.array(Funcframe)
#    cA2 = cA2.astype(np.uint8)
    Funcframe = Funcframe.astype(np.uint8)
    
#    coeffs = wavedec2(Funcframe, 'haar', level=1)   
#    cA2 = coeffs[1][0]
#    FuncframeFeature = cA2
    
    ###############################
    ###HOG FEATURE EXTRACTION
    ###############################
    img = Funcframe
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    FuncframeFeature = hog.compute(img)\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
# hog_feats now contains the gradient amplitudes for each direction,
# for each cell of its group for each group. Indexing is by rows then columns.
    
    FuncframeFeature = FuncframeFeature.flatten()

    
    return FuncframeFeature

########################################
### CLUSTERING ALGORITHM 
########################################
def performClustering(frameFeat, frameNumber, k, beta, centroids, idx):     
    samples = frameFeat 
    #samples = samples.flatten()
    N_samples = 1
    N_centroids = k;
    z = np.random.rand(1, N_samples)
    print(frameNumber)
    #iter_labels = np.zeros(N_samples)
    div_cost = np.zeros(N_samples)
    for i in range(0, N_samples):
        C = 1/float(10*(frameNumber+1))
        if frameNumber<N_centroids:
            centroids.append(samples)
            idx.append(frameNumber)
        else:
            #np.array(centroids)
            maxdiv = CentroidsDivScore(centroids)
            X = samples
            dist = np.zeros(int(N_centroids))
            for j in range(0,int(N_centroids)):
                new_centroids = list(centroids)
                new_centroids[j] = X;                
                div1 = CentroidsDivScore(new_centroids)
                div = div1-maxdiv
                dist[j] = beta*(np.linalg.norm(np.array(X) - np.array(centroids[j]))) - (1-beta)*C*(div)
            best_idx = np.argmin(dist)
            #iter_labels[i] = best_idx
            W_best_old = centroids[best_idx]
            idx_old = idx[best_idx]
            centroids[best_idx] = X
            idx[best_idx] = frameNumber
            new_cost = CentroidsDivScore(centroids)
            
            if (new_cost>=maxdiv or z[0][i]<0.01): #add noise so that a particular cluster doesnt win too often
                maxdiv = new_cost
            else:
                centroids[best_idx] = W_best_old
                idx[best_idx] = idx_old
            div_cost[i] = new_cost
    #print("--- RUN TIME %s seconds --- \n" % (time.clock() - start_time))
    return centroids, idx
#######################################
##CALCULATE DIVERSITY SCORE
#######################################
def CentroidsDivScore(centroids1):
    C11 = np.array(centroids1)
    mean_vec = np.mean(C11, axis=0)
    dist = 0
    for s in range(0, C11.shape[0]):
        row = C11[s,:]
        p = row-(mean_vec)
        norm_of_vec = np.linalg.norm(p)
        dist += norm_of_vec
    div_score = dist
    return div_score

########################################
### MAIN CODE
########################################
print("Hello :D")
my_in=0
while not(my_in==1 or my_in==2):
    my_in = input("Enter 1 to run the code on VSUMM dataset and 2 to run code on webcam data")

#beta_array = np.arange(0,1.2,0.2)
beta_array = np.array([0])
for beta in np.nditer(beta_array):
    if my_in==1:
        path = "./Videos/database/"
        vid_names = sorted(os.listdir(path))
        num_vid = np.arange(0,len(vid_names))
        string = "dataset"
    else:
        num_vid = np.array([1]) #uncomment this for webcam streaming video. 
        string = "webcam"
    for vidNum in np.nditer(num_vid):
        if my_in==1:
            video = vid_names[vidNum]
            saveFolder = "./Videos/results_HOG_"+string+"/python_summary_"+str(beta)+"/"+video[0:3]
            video = vid_names[vidNum]
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            frameNum=0
            exemplarIndices = list()
            saveFileName = "pythonSummary_"+video[0:3]
            ##################################
            ##EXTRACT NUMBER OF SUMMARY 
            ##FRAMES FROM USER GIVEN SUMMARIES
            ##################################
    
            path2 = "./Videos/UserSummary/"+video[0:3] #to remove the .mpg extension from the name
            N = np.zeros((1, len(os.listdir(path2))))
            for s in range(0, len(os.listdir(path2))):
                N[0][s] = len(os.listdir(path2 + "/user"+str(s+1)))
            numSummary = max(N[0])
            if numSummary < 5:
                numSummary = numSummary * 2
            print("Number of Summary frames will be ",numSummary)
            
        else:
            saveFolder = "./Videos/results_HOG_"+string+"/python_summary_"+str(beta)+"/"
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            numSummary = 8 #set it to whatever you want
            print("Number of Summary frames will be ",numSummary) 
            index1 = list()
            index2 = list()
            centroidFrames = list()
            frameNum=0
            
    
        ##################################
        ##OPEN THE VIDEO
        ##################################
        if my_in==1:
            cap = cv2.VideoCapture(path+video)
            condition = cap.isOpened()
        else:
            cap = cv2.VideoCapture(0)  #this is open webcam and stream data from it. 
            begin_time=time.time()
            condition = (time.time() - begin_time <120) #set time as desired; here it is 120 seconds
        print("Frame rate is ",cap.get(cv2.CAP_PROP_FPS))
        print("Video opened is ",cap.isOpened())
        exemplarIndices = list()
        index = list()
        start_time = time.clock()
        while(condition):
            ret, frame = cap.read()
            if ret is True:     
                cv2.imshow('frame',frame)
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #####################################
                ##FEATURE EXTRACTION
                #####################################
                frameFeature = featureExtraction(grayFrame)
            
                #####################################
                ##CLUSTERING
                #####################################  
                
                if my_in==2:
                    exemplarIndices,index2 = performClustering(frameFeature, frameNum, int(numSummary), beta, exemplarIndices, list(index1))
                    if frameNum<numSummary:
                        centroidFrames.append(frame)   
                    else:
                        val1 = list(set(index1) - set(index2))
                        if val1:
                            changedFrame = index1.index(val1[0])
                            centroidFrames[changedFrame] = frame
                    p = list(index2)    
                    index1 = p    
                    condition = (time.time() - begin_time <120)
#                    if cv2.waitKey(1) & 0xFF == ord('q'):
#                        print("Here")
#                        index1 = np.sort(index1)
#                        for saveIndex in range(0, len(centroidFrames)):
#                            name = saveFolder+str(index1[saveIndex])+".jpg"
#                            cv2.imwrite(name,centroidFrames[saveIndex])                    
#                        break
                    
                    shutil.rmtree(saveFolder)
                    os.makedirs(saveFolder)
                    index_final = np.sort(index1)
                    for saveIndex in range(0, len(centroidFrames)):
                        name = saveFolder+str(index_final[saveIndex])+".jpg"
                        cv2.imwrite(name,centroidFrames[saveIndex])                    
#                    break
                    
                else:
                    exemplarIndices,index = performClustering(frameFeature, frameNum, int(numSummary), beta, exemplarIndices, index)
                    condition = cap.isOpened()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                print frameNum
                frameNum+=1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("--- RUN TIME %s seconds --- \n" % (time.clock() - start_time))
        print("Processing speed is",frameNum/(time.clock() - start_time))
        print("Out of while loop")
        sorted_idx = np.sort(index)
        print(sorted_idx)
        #sorted_idx1 = {"summary_frames_py" : sorted_idx}
    
        #####################################
        ##SAVE SUMMARY FRAMES
        #####################################
        if my_in==1:
            cap1 = cv2.VideoCapture(path+video)
            num=0
            while(cap1.isOpened()):
                ret, frame = cap1.read()
                if ret is True:
                    if num in sorted_idx:
                        name = saveFolder+"/"+saveFileName+"_"+str(num)+".jpg"
                        cv2.imwrite(name,frame)
                        num+=1
                    else:
                        num+=1
                        continue
                else:
                    break
            cap1.release()
            cv2.destroyAllWindows()
            print("Done with Video ",vidNum)


########################################
### DO ONLINE CLUSTERING WITH SOME DIVERSITY SCORE
########################################
