# EECS 435 Final Project:Wetland Image Recognition

This repo is the final project of EECS_435 DL:FAA at Northwestern.

The whole project is implemeted and tested with Google Colab.

Contributer: Maorong Wang and Yuanzhe Jin

Due to NAIP dataset is too large for github to handle, you can access the dataset [here](https://drive.google.com/open?id=1rzYHHWeeFoArQtmOFaYIB29Gzpo4qUjO).

The easiest way to set this up:

1.  Upload jupyter notebook file and three .py file to your Google Colaboratory.
2.  Upload the dataset on your own Google Drive.  
    Your directory should appear like this in Google Drive:
    <pre>
    ../  
        /dataset  
          /Nonwetland.mat  
          /wetland.mat  
        /WetlandImageRecognition.ipynb  
        /LeNet5.py  
        /MyNet.py  
        /Dataloader.py  
    </pre>
3.  Mount Google Drive in the jupyter notebook and execute the cells.

If you wish to download the code and dataset on your own PC and test the code, please rememeber to adjust `root_path` and `data_loader.py`.
