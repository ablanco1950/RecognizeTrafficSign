import os
import shutil
output_dir="Dir_TrafficSign_Resnet"
Num_clases=43
if  os.path.exists(output_dir):shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + "\\train")
os.mkdir(output_dir+ "\\valid")
#os.mkdir(output_dir + "\\test")
for i in range (Num_clases):
    if len(str(i)) < 2:
       NameDir="0" + str(i)
    else:   
       NameDir= str(i)
    os.mkdir(output_dir + "\\train\\"+NameDir)
    os.mkdir(output_dir+ "\\valid\\"+NameDir)
