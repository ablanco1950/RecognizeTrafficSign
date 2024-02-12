# RecognizeTrafficSign.

From traffic sign database downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
It is produced, using a CNN based on resnet and pytorch, a model to recognize traffic signs.

In the images, the traffic signs are detected using the model generated at: https://github.com/ablanco1950/DetectTrafficSign before recognition.

Model resnet pytorch has been chosen due to the good results it provided in the project https://github.com/ablanco1950/CarsModels_Resnet_Pytorch

Creation of the necessary file structure from the dataset downloaded from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign which is supposed to appear as archive (14).zip,(inn practice only 39,209 records can be downloaded for train and the decompressor seems to stop at the end). Once decompressed , execute:

Create_DirTrafficSign_Resnet.py

which creates the data structure needed for resnet pytorch

Then fill in that data structure by executing:

Fill_DirTrafficSign_Resnet.py

Train the model by running:

Train120x120_RecognizeTrafficSign_Resnet_Pytorch.py

which produces the checkpoint120x120_10epoch.pth model, due to its size, i have not been able to upload it to github.

  The log of this training is attached in

  LOG_Train120x120_10epoch.txt

in which it is indicated:

Test accuracy of model: 99.435%

which is far from the results when applied in real cases.

You can then execute:

Guess120x120TrafficSign_Resnet_Pytorch.py

which would give a success rate of 99.43502824858757%

Next, it is tested with a series of photos that appear in the Test2 folder ( formed with the first 30 images from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data.

running:

RecognizetTrafficSign120x120_10epoch_Detected_yolo_resnet_pytorch.py

The detected signals are presented and the recognized name of the detected signal is displayed on the console and at the end of the detected signals in each image, the image with the detected and recognized images.
There are errors in any signal in images: 00009.jpg, 00011.jpg and 00022.jpg

Using the model (model.h5) downloaded from https://github.com/AvishkaSandeepa/Traffic-Signs-Recognition and testing with the same images by executing :

RecognizetTrafficSignDetectedAvishkaSandeepa.py

appears errors in signals in images:00011.jpg, 00012.jpg, 00017.jpg, 00019.jpg y 00025.jpg

Observation:

The results when images with the same cfeatures as those downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data are not used are much worse and differ depending on the model or within each train of the same model according to the epoch used. Therefore, it would be necessary to use a specific model for each signal or group of signals, which would complicate the treatment or, an ignored treatment of the images so that they had the same features as those downloaded from https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format/data

