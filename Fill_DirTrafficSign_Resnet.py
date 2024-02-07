# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , jan 2024
"""

######################################################################
# PARAMETERS
######################################################################
Factor_split=0.995
From_dirname = "archive (14)\\Train"

######################################################################

import os
import re

import cv2

import numpy as np

imgpath = From_dirname + "\\"

print("Reading imagenes from ",imgpath)

TabDirName=[]
for root, dirnames, filenames in os.walk(imgpath):
 for dirname in dirnames:  
    #print(dirname)
    TabDirName.append(dirname)
    
for i in range(len(TabDirName)):
    
    imgpath1=imgpath+ str(TabDirName[i])+"\\"

    TotImages=0
    TotImagesTrain=0
    TotImagesValid=0
    images = []
    NameImages=[]
    # https://stackoverflow.com/questions/62137343/how-to-get-full-path-with-os-walk-function-in-python
    for root, dirnames, filenames in os.walk(imgpath1):
      for filename in filenames:  
        #print(filename)
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            
            filepath = os.path.join(root, filename)
            # https://stackoverflow.com/questions/51810407/convert-image-into-1d-array-in-python
            
            image = cv2.imread(filepath)
            #cv2.imshow("image",image)
            #cv2.waitKey(0)
            images.append(image)
            NameImages.append(filename)
                       
            TotImages+=1
            
    
    if len(TabDirName[i]) < 2:
        dirnameTo= "0" + TabDirName[i]
    else:
        dirnameTo=  TabDirName[i]
    limit=int(len(images)*Factor_split)
    for j in range(len(images)):
        if j > limit:
            cv2.imwrite("Dir_TrafficSign_Resnet\\valid\\" + dirnameTo + "\\" + NameImages[j], images[j])
            TotImagesValid+=1
        else:    
            cv2.imwrite("Dir_TrafficSign_Resnet\\train\\" + dirnameTo + "\\" + NameImages[j], images[j])
            TotImagesTrain+=1
    print( dirnameTo + " has " + str(len(images)) + " in train " + str(TotImagesTrain)  +  " in valid " + str(TotImagesValid))     

