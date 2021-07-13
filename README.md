# PMGI
An unofficial implementation of Paper "Rethinking the Image Fusion: A Fast Unified Image Fusion Network based on Proportional Maintenance of Gradient and Intensity" using PyTorch

## NOTICE  
1.The metric SSIM remains some problems to be solve.  
2.When using this code for the first time, please create a folder under pwd and name it as 'pth' to save model.  
3.Prograss visualization is available when training, if you train the model on a server, please delete relevent part.  
4.Caution device type in each file ('cpu' or 'cuda:0')

## HOW TO USE
`python train.py` for training  

## ENVIRONMENT
torch==1.7.1+cu110  
torchvision==0.8.2+cu110  
scipy>=1.3.0  
imageio>=2.9.0  
numpy>=1.19.5  
