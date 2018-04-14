

Team    : MASS
Github Repo :   https://github.com/sudheerachary/Manga_Colorization.git

cGAN colorization 
FOLDER NAME:Models
consists the GAN models and visualizations.
train.py    :   run to train on images stored in Datasets/Train
test.py     :   run to generate colorization for images in Datasets/Test

Folder NAME :  Datasets
Generated_Images    : contains generated images and trained model weights.
Test        :   Put testing images here
Train       :   Put new dataset here
Validation  :   Validation set while training.

Segmentation and post-processing

FOLDER NAME:segmentation_post_process

Folder-Description
input_images:   contains test images						(numbered from 1-8 in jpeg format)
bw_images   :	contains the black and white images after screentone removal	(numbered from 1-8 in jpg format)
gan_images  :   contains gan output						(numbered from 1-8 in jpg format)
Results     :   contains results for postprocessing and final outputs

Code:
segmentation_final.m  : Code written in matlab 
Run segmentation_final.m in MATLAB.

Results : Will be stored in results folder (contains segmented,kmeans,final outputs).

To test For a particular image, 
1)Rename the output of GAN as any number between 1-8 in .jpg and place it in gan_images.
2)Put the gray scale image in input_images. 
3)Run segmentation_final.m