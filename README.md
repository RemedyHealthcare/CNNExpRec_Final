This code trains a convolutional neural network on the fer2013 dataset, and analyzes it too.

Important: the data to run this code is too large for github. Please download it from here:

https://drive.google.com/file/d/0ByNRvvJIT4L8Si0zY3FudVV6NG8/view

And unzip it inside this directory.

CNN.py Trains a network, and produces the file 'Trained_CNN-xx' where xx is the last epoch the network completed.
To adjust train and test parameters before executing the code, simply open it in your text editor of choice, and adjust the parameters at the top of the file.

CNN_eval.py Evaluates a trained network. To adjust train and test parameters before executing the code, as well as the saved network to evaluate, simply open it in your text editor of choice, and adjust the parameters at the top of the file. 

Training a network takes a lot of time, and I'd recommend using a GPU to do it. Tensorflow is realtively new and doesn't play nicely with all GPUs, but I've found using an AWS g2.2xlarge instance running community AMI ami-cf5028a5 will do the trick.

You can download a pretrained network here: 

https://drive.google.com/file/d/0ByNRvvJIT4L8WXVadm9KOW90SzQ/view?usp=sharing

Place this file inside this directory.

To run this code, you will need to have TensorFlow, scipy, numpy, and matplotlib installed

