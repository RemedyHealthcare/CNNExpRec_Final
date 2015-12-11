import numpy as np
import os
#import matplotlib.pyplot as plt
import tensorflow as tf
#import matplotlib.image as mpimg
class Fer_Parser:


    def parse_all(self):
        print 'Parsing all data...'
        f = open('data/fer2013/fer2013.csv')
        X_tr = []
        Y_tr = []
        X_te = []
        Y_te = []

        count = 0
        lines = f.readlines()[1:]
        length = str(len(lines))
        for line in lines:
            count += 1
            if count % 5000 == 0:
                print str(count) + '/' +length
            data = line.split(',')
            tag = int(data[0])

            pixels = map(int, data[1].split(' '))
            pixel_holder = np.array(pixels)

            pixels = list(np.reshape(pixel_holder, (48,48)))
            data_set = data[2][:-1]



            if data_set == 'Training':

                X_tr.append(pixels)
                Y_tr.append(tag)

            if data_set == 'PublicTest':
                X_te.append(pixels)
                Y_te.append(tag)

        #convert Y to one hot
        X_tr = np.array(X_tr)
	X_tr = X_tr - np.mean(X_tr, axis = 0)
        X_tr = X_tr.reshape(np.shape(X_tr)[0], 48, 48, 1)
        Y_tr = np.eye(7)[np.array(Y_tr, dtype = np.uint8)]
        X_te = np.array(X_te)
	X_te = X_te - np.mean(X_te, axis = 0)
        X_te = X_te.reshape(np.shape(X_te)[0], 48, 48, 1)
        Y_te = np.eye(7)[np.array(Y_te, dtype = np.uint8)]
        print 'Done parsing data.'
        return X_tr, Y_tr, X_te, Y_te



'''
#Example
p = Fer_Parser()
a,b,c,d = p.parse_all()
'''
