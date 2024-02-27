# RecognizeTrafficSign.

From traffic sign database downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
It is produced, using a CNN based on resnet and pytorch, a model to recognize traffic signs.

In the images, the traffic signs are detected using the model generated at: https://github.com/ablanco1950/DetectTrafficSign before recognition.

Model resnet pytorch has been chosen due to the good results it provided in the project https://github.com/ablanco1950/CarsModels_Resnet_Pytorch

Creation of the necessary file structure from the dataset downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign which is supposed to appear as archive (14).zip,(in practice only 39,209 records can be downloaded for train and the decompressor seems to stop at the end). Once decompressed , execute:

Create_DirTrafficSign_Resnet.py

which creates the data structure needed for resnet pytorch

Then fill in that data structure by executing:

Fill_DirTrafficSign_Resnet.py

Train the model by running:

Train224x224_RecognizeTrafficSign_Resnet_Pytorch.py

which produces the checkpoint224x224_3epoch.pth model, due to its size, i have not been able to upload it to github.

The log of this training is attached in

 LOG_Train224x224_3epoch.txt

in which it is indicated:

Test accuracy of model: 98.875%

You can then execute:

Guess224x224TrafficSign_Resnet_Pytorch.py

which would give a success rate of 100% (only 177 images tested)

Next, it is tested with a series of photos that appear in the Test2 folder ( formed with the first 30 images from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data.)

 (the attached Test2.zip folder must be decompressed and you must ensure that there is only one Test2 folder and not 2 nested ones as the decompressor usually does)

Execute:

RecognizetTrafficSign224x224_3epoch_Detected_yolo_resnet_pytorch.py

The detected signals are presented and the recognized name of the detected signal is displayed on the console and at the end of the detected signals in each image, the image with the detected and recognized images.
There are errors in any signal in images: 00009.jpg, 00016.jpg, 00018.jpg and 00019.jpg

Using the model (model.h5) downloaded from https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition and testing with the same images by executing :

RecognizetTrafficSignDetectedAvishkaSandeepa.py

appears errors in signals in images:00011.jpg, 00012.jpg, 00017.jpg, 00019.jpg y 00025.jpg

Observation:

The results with images with distinct features as those downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data are  worst and differ depending on the model was trained.

For example testing the images in Test folder (attched compressed  in this project) by changing in  11 line of RecognizetTrafficSign224x224_3epoch_Detected_yolo_resnet_pytorch.py Test2 by Test, the results are worst. The same in RecognizetTrafficSignDetectedAvishkaSandeepa.py

After seen that  (model.h5) downloaded from https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition is better recognizing speed limits   signals and in the rest is better checkpoint224x224_3epoch.pth, is created :

RecognizetTrafficSign_SeveralModels.py 

that uses the two models: model.h5 and checkpoint224x224_3epoch.pth getting the best accuracy

