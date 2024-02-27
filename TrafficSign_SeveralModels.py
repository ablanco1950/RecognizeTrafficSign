
import cv2
import numpy as np

from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image

import torch
from torch import nn

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

#model = models.resnet34(pretrained=True)
model = models.resnet50(pretrained=True)

# https://stackoverflow.com/questions/53612835/size-mismatch-for-fc-bias-and-fc-weight-in-pytorch
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 43)

#TabCarBrand=[]
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#model_path= "checkpoint20epoch.pth"
#model_path60x60="checkpoint60x60_20epoch.pth"
#model_path20x20="checkpoint20x20_7epoch.pth"
model_path= "checkpoint224x224_3epoch.pth"

from keras.models import Sequential, load_model
# Downloaded from https://github.com/faeya/traffic-sign-classification
#modelTrafficSignRecognition=load_model('traffic_classifier.h5')

# downloaded from https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition
# 
modelTrafficSignRecognition=load_model('model.h5')
def process_image(image,  option_transform):
    
    # Process a PIL image for use in a PyTorch model
  
    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}')

    transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    """ 
    if  option_transform==1:
        
        transform = transforms.Compose([transforms.Resize((20,20)),
                                            transforms.CenterCrop(20),
                                            transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    """  


    # Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def predict_TrafficSign(image_path, model, topk,  option_transform):
    topk=5
    # Implement the code to predict the class from an image file   
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    img = process_image(image_path,  option_transform)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted

def PredictAvishkaSandeepa(gray1):
                data=[]
                # for trafficclasifier
                #SignalToRecognize=cv2.resize(gray1,(30,30))
                # for https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition
                SignalToRecognize=cv2.resize(gray1,(32,32))
                data.append(np.array(SignalToRecognize))
                X_test=np.array(data)
                predict_x=modelTrafficSignRecognition.predict(X_test) 
                classes_x=np.argmax(predict_x,axis=1)
                NameTrafficSignPredicted_model2=str(classes[int(classes_x)+1])
                return NameTrafficSignPredicted_model2

def ProcessTrafficSign_SeveralModels(gray1):
                topk=5
                NameTrafficSignPredicted=""
                cv2.imwrite("pp.jpg",gray1)
                option_transform=0
                conf1, predicted1=predict_TrafficSign("pp.jpg", model_path, topk,  option_transform)
                #NameTrafficSignPredicted_model1=TabTrafficSign[predicted1[0]]+ " "+ str(conf)
                NameTrafficSignPredicted_model1=TabTrafficSign[predicted1[0]]
                # in signals speed limit PredictAvishkaSandeepa has more accuracy
                if NameTrafficSignPredicted_model1 == 'Speed limit (20km/h)':
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                elif NameTrafficSignPredicted_model1 == 'Speed limit (30km/h)':
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                elif   NameTrafficSignPredicted_model1 =='Speed limit (50km/h)':
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                elif  NameTrafficSignPredicted_model1 =='Speed limit (60km/h)':
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)                   
                elif   NameTrafficSignPredicted_model1 =='Speed limit (70km/h)' :                        
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                elif   NameTrafficSignPredicted_model1 =='Speed limit (80km/h)' :                        
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
               
                elif   NameTrafficSignPredicted_model1 =='End of speed limit (80km/h)' :                        
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                elif   NameTrafficSignPredicted_model1 =='Speed limit (100km/h)' :                        
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                          
                elif   NameTrafficSignPredicted_model1 =='Speed limit (100km/h)' :                        
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                          
                elif   NameTrafficSignPredicted_model1 == 'Speed limit (120km/h)' :
                          NameTrafficSignPredicted=PredictAvishkaSandeepa(gray1)
                else:
                      NameTrafficSignPredicted=NameTrafficSignPredicted_model1

                """
                option_transform=1
                conf0, predicted0=predict_TrafficSign("pp.jpg", model_path20x20, topk,option_transform)
                #NameTrafficSignPredicted_model1=TabTrafficSign[predicted1[0]]+ " "+ str(conf)
                NameTrafficSignPredicted_model0=TabTrafficSign[predicted0[0]]
                option_transform=0
                conf1, predicted1=predict_TrafficSign("pp.jpg", model_path, topk,  option_transform)
                #NameTrafficSignPredicted_model1=TabTrafficSign[predicted1[0]]+ " "+ str(conf)
                NameTrafficSignPredicted_model1=TabTrafficSign[predicted1[0]]
                
                if    NameTrafficSignPredicted_model0 =='Roundabout mandatory' :
                        NameTrafficSignPredicted=NameTrafficSignPredicted_model0 + " " + str(conf0[0])

                elif NameTrafficSignPredicted_model1 == 'Stop' or  NameTrafficSignPredicted_model1 =='No passing' or \
                    NameTrafficSignPredicted_model1 =='Yield' or \
                    NameTrafficSignPredicted_model1 =='Bumpy road' or \
                    NameTrafficSignPredicted_model1 =='Turn right ahead' or \
                    NameTrafficSignPredicted_model1 =='Keep right' or \
                    NameTrafficSignPredicted_model1 =='No entry' :
                        NameTrafficSignPredicted=NameTrafficSignPredicted_model1 +  " " + str(conf1[0])

                else:
                        data=[]
                        # for trafficclasifier
                        #SignalToRecognize=cv2.resize(gray1,(30,30))
                        # for https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition
                        SignalToRecognize=cv2.resize(gray1,(32,32))
                        data.append(np.array(SignalToRecognize))
                        X_test=np.array(data)
                        predict_x=modelTrafficSignRecognition.predict(X_test) 
                        classes_x=np.argmax(predict_x,axis=1)
                        NameTrafficSignPredicted_model2=str(classes[int(classes_x)+1])
                        #print(str(classes[int(classes_x)+1]))
                        if NameTrafficSignPredicted_model2 == 'Speed limit (20km/h)':
                          NameTrafficSignPredicted=NameTrafficSignPredicted_model2
                        if NameTrafficSignPredicted_model2 == 'Speed limit (30km/h)':
                          NameTrafficSignPredicted=NameTrafficSignPredicted_model2
                        if   NameTrafficSignPredicted_model2 =='Speed limit (50km/h)':
                              NameTrafficSignPredicted=NameTrafficSignPredicted_model2
                        if  NameTrafficSignPredicted_model2 =='Speed limit (60km/h)':
                             NameTrafficSignPredicted=NameTrafficSignPredicted_model2                   
                        if   NameTrafficSignPredicted_model2 =='Speed limit (70km/h)' :                        
                               NameTrafficSignPredicted=NameTrafficSignPredicted_model2
                        if   NameTrafficSignPredicted_model2 =='Speed limit (80km/h)' :                        
                               NameTrafficSignPredicted=NameTrafficSignPredicted_model2
                        if   NameTrafficSignPredicted_model2 =='End of speed limit (80km/h)' :                        
                               NameTrafficSignPredicted=NameTrafficSignPredicted_model2       
                        
                        if   NameTrafficSignPredicted_model2 == 'Bicycles crossing' :
                                    NameTrafficSignPredicted=NameTrafficSignPredicted_model2
                """
                return NameTrafficSignPredicted
                 
