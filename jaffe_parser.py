import numpy as np
import os
#import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imresize
import matplotlib.image as mpimg
class Jaffee_Parser:
    def images_to_tensor(self):
        images = []

        for file in sorted(os.listdir('data/jaffe_images_small')) :

            if(file != '.DS_Store'):

                image = mpimg.imread('data/jaffe_images_small/' + file)
                if (len(np.shape(image)) > 2):
                    image = image[:,:,0]
                image = image.tolist()
                image = imresize(image, (48,48))

                images.append(image)

        image_tensor = np.array(images)
        image_tensor = image_tensor - np.mean(image_tensor, axis = 0)

        image_tensor = image_tensor.reshape(213, 48, 48, 1)

        return image_tensor

    def text_to_tensor(self):
        labels = []
        text = open('data/jaffe_labels/JAFFE_labels.txt')
        for line in text.readlines()[2:]:
            line_labels = line.split()
            labels.append(line_labels[1:-1])
        label_tensor = np.array(labels)

        return label_tensor

    def text_to_one_hot(self):
        #labels are of the form [NEU HAP SAD SUR ANG DIS FEA]
        labels = []
        text = open('data/jaffe_labels/JAFFE_labels.txt')
        for line in text.readlines()[2:]:
            line_label = line[-4:-2]
            tag = line [-7:-1]

            index = 0
            if line_label == 'NE':
                index == 6
            if line_label == 'HA':
                index = 3
            if line_label == 'SA':
                index = 4
            if line_label == 'SU':
                index = 5
            if line_label == 'AN':
                index = 0
            if line_label == 'DI':
                index = 1
            if line_label == 'FE':
                index = 2
            labels.append([tag,index])

        label_tensor = np.array(labels)

        i  = np.argsort(label_tensor[:,0])

        label_tensor = label_tensor[i]

        label_tensor = label_tensor[:,1]

        label_tensor = label_tensor.astype(int)

        one_hot = np.zeros([np.size(label_tensor), 7])
        for i in range(np.size(label_tensor)):
            one_hot[i, label_tensor[i]] = 1
        return one_hot
