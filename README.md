# RecognizeTrafficSign.

From traffic sign database downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
It is produced, using a CNN based on resnet and pytorch, a model to recognize traffic signs.
As the results are not satisfactory for all traffic signs, it is combined with other models by varying the model parameters or using other CNN models.

In the images, the traffic signs are detected using the model generated at: https://github.com/ablanco1950/DetectTrafficSign before recognition.

Creation of the necessary file structure from the dataset downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign which is supposed to appear as archive (14).zip, once decompressed , execute:

Create_DirTrafficSign_Resnet.py

which creates the data structure needed for resnet pytorch

Then fill in that data structure by executing:

Fill_DirTrafficSign_Resnet.py

Train the model by running:

Train_RecognizeTrafficSign_Resnet_Pytorch.py

which produces the checkpoint20epoch.pth model, due to its size I have not been able to upload it to github.

  The log of this training is attached in

  LOG_Train_RecognizeTrafficSign_Resnet_Pytorch.txt

in which it is indicated:

Test accuracy of model: 94.35%

which is far from the results when applied in real cases.

You can then execute:

GuessTrafficSign_Resnet_Pytorch.py

  which would give a success rate of 98.3%

Next, it is tested with a series of photos that appear in the Test folder and using the one obtained in the project as a traffic sign detector:

https://github.com/ablanco1950/DetectTrafficSign


running:

RecognizetTrafficSignDetected_yolo_resnet_pytorch.py

The detected signals are presented and the recognized name of the detected signal is displayed on the console and at the end of the detected signals in each image, the image with the detected and recognized images.

The result is that the model seems to recognize the signals well:

   'Stop' ,'No passing' ,'Yield' , 'Bumpy road' ,'Turn right ahead' , 'Keep right' and 'No entry'

On the other hand, in the project https://github.com/faeya/traffic-sign-classification using the traffic_classifier.h5 model (you have to download it from the project). It is noted that the speed limit and 'Bicycles crossing' signs

That is, even using the same training data, depending on the model, some signals or others are better recognized.

Even using the same resnet pytorch model and generating the checkpoint20x20_7epoch.pht model, after executing:

Train20x20_RecognizeTrafficSign_Resnet_Pytorch.py it is verified that with the same model and train base, changing only the epoch and the dimension of the box, it is possible to correctly identify the Roundabout mandatory sign.

The different models are combined and improved recognition is obtained, running


RecognizetTrafficSign_SeveralModels.py


and the video version

VIDEORecognizeTrafficSign_SeveralModels.py

It is expected to incorporate more models that allow improving this image recognition.
