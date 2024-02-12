# -*- coding: utf-8 -*-
"""
Created on feb 2024

@author: Alfonso Blanco using the model downloaded from
https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition
"""
#######################################################################
# PARAMETERS
######################################################################
dir=""
dirname= "Test2"
#dirnameYolo="runs\\detect\\train2\\weights\\best.pt"
dirnameYolo="bestDetectTrafficSign.pt"

from keras.models import Sequential, load_model
# Downloaded from https://github.com/faeya/traffic-sign-classification
#modelTrafficSignRecognition=load_model('traffic_classifier.h5')

# https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition
modelTrafficSignRecognition=load_model('model.h5')


#creating dictionary to label all traffic sign classes.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }


import cv2
import time

TimeIni=time.time()



# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model_yolo = YOLO(dirnameYolo)
class_list = model_yolo.model.names
print(class_list)
TabTrafficSign =[ 'Speed limit (20km/h)',
            'Speed limit (30km/h)',      
            'Speed limit (50km/h)',       
            'Speed limit (60km/h)',      
            'Speed limit (70km/h)',    
            'Speed limit (80km/h)',      
            'End of speed limit (80km/h)',     
            'Speed limit (100km/h)',    
            'Speed limit (120km/h)',     
            'No passing',   
            'No passing veh over 3.5 tons',     
            'Right-of-way at intersection',     
            'Priority road',    
            'Yield',     
            'Stop',       
            'No vehicles',       
            'Veh > 3.5 tons prohibited',       
            'No entry',       
            'General caution',     
            'Dangerous curve left',      
            'Dangerous curve right',   
            'Double curve',      
            'Bumpy road',     
            'Slippery road',       
            'Road narrows on the right',  
            'Road work',    
            'Traffic signals',      
            'Pedestrians',     
            'Children crossing',     
            'Bicycles crossing',       
            'Beware of ice/snow',
            'Wild animals crossing',      
            'End speed + passing limits',      
            'Turn right ahead',     
            'Turn left ahead',       
            'Ahead only',      
            'Go straight or right',      
            'Go straight or left',      
            'Keep right',     
            'Keep left',      
            'Roundabout mandatory',     
            'End of no passing',      
            'End no passing veh > 3.5 tons' ]
import torch
from torch import nn
import os
import re

import cv2

import numpy as np
import keras
import functools  
import time
inicio=time.time()

import numpy as np

import numpy



########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                                            
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectTrafficSignWithYolov8 (img):
  
   TabcropTrafficSign=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   results = model_yolo.predict(img)
   for i in range(len(results)):
       # may be several plates in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       print(class_id)
       out_image = img.copy()
       for j in range(len(class_id)):
           con=confidence[j]
           label=class_list[class_id[j]] + " " + str(con)
           box=xyxy[j]
           
           cropTrafficSign=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           TabcropTrafficSign.append(cropTrafficSign)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))
           #Tabclass_name.append(class_name)
           print(label)
           Tabclass_name.append(label)
            
      
   return TabcropTrafficSign, y,yMax,x,xMax, Tabclass_name



###########################################################
# MAIN
##########################################################

imagesComplete, TabFileName=loadimages(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

ContDetected=0
ContNoDetected=0

for i in range (len(imagesComplete)):
          
            gray=imagesComplete[i]

            #cv2.imshow('Gray', gray)
            #cv2.waitKey(0)
            
            TabImgSelect, y, yMax, x, xMax, Tabclass_name =DetectTrafficSignWithYolov8(gray)
            
            if TabImgSelect==[]:
                print(TabFileName[i] + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                ContDetected=ContDetected+1
                print(TabFileName[i] + " DETECTED ")
            for z in range(len(TabImgSelect)):
                #if TabImgSelect[z] == []: continue
                gray1=TabImgSelect[z]
                #cv2.waitKey(0)
                start_point=(x[z],y[z]) 
                end_point=(xMax[z], yMax[z])
                color=(0,0,255)
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 5 px
                img = cv2.rectangle(gray, start_point, end_point,(255,0,0), 5)
                # Put text
                text_location = (x[z], y[z])
                text_color = (255,255,255)

                
                cv2.imwrite("pp.jpg",gray1)
                data=[]
                SignalToRecognize=cv2.resize(gray1,(32,32))
                data.append(np.array(SignalToRecognize))
                X_test=np.array(data)
                predict_x=modelTrafficSignRecognition.predict(X_test) 
                classes_x=np.argmax(predict_x,axis=1)
                NameTrafficSignPredicted_model2=str(classes[int(classes_x)+1])
                print(NameTrafficSignPredicted_model2)
                cv2.putText(img, NameTrafficSignPredicted_model2 ,text_location
                   , cv2.FONT_HERSHEY_SIMPLEX , 1
                   , text_color, 2 ,cv2.LINE_AA)
                cv2.putText(gray1, NameTrafficSignPredicted_model2 ,text_location
                   , cv2.FONT_HERSHEY_SIMPLEX , 1
                   , text_color, 2 ,cv2.LINE_AA)       
                cv2.imshow('Trafic Sign', gray1)
                cv2.waitKey(0)
            #      
            show_image=cv2.resize(img,(1000,700))
            cv2.imshow('Frame', show_image)
            cv2.waitKey(0)
           
             
              
print("")           

print( " Time in seconds "+ str(time.time()-TimeIni))
