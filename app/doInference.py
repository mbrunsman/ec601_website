#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

import cv2
import matplotlib.pyplot as plt
import copy
import os


# In[3]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# In[40]:


def readImageAsArray(imagePath):
    img = cv2.imread(imagePath)
    return img

def showImage(imageArray):
    # plots image array as colored image 
    plt.imshow(cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB))


def makeInference(filename):
    print("called to make inference\n")
    model = keras.models.load_model("model-save.h5", 
                                custom_objects={'bce_dice_loss': bce_dice_loss,
                                                'dice_coef': dice_coef,
                                               })

    test_img_path = os.getcwd() + "/static/img/" + filename + ".jpg"
    print("image path: ")
    print(test_img_path)
    # test_img_path = "/app/static/img/00dbd3c.jpg"
    imageAsArray = readImageAsArray(test_img_path)    
    imageAsArrayCropped = cv2.resize(imageAsArray, (480, 320))   
    imageAsArrayExpanded = np.expand_dims(imageAsArrayCropped, axis=0)
    modelPrediction = model.predict(imageAsArrayExpanded)
    predictionMasks = modelPrediction[0, ].round().astype(int)

    
    maskedImage = copy.deepcopy(imageAsArrayCropped)
    print(maskedImage.shape)

    for row in range(320):
        for col in range(480):
            if(predictionMasks[row][col][0]==1):
                maskedImage[row][col] = [255,0,0]
            elif(predictionMasks[row][col][1]==1):
                maskedImage[row][col] = [0,255,0]
            elif(predictionMasks[row][col][2]==1):
                maskedImage[row][col] = [0,0,255]
            elif(predictionMasks[row][col][3]==1):
                maskedImage[row][col] = [100,100,100]
            else:
                maskedImage[row][col] = [0,0,0]

    
    plt.imshow(imageAsArrayCropped, cmap='gray')
    plt.imshow(maskedImage, cmap='gist_rainbow_r', alpha=0.5)
    plt.savefig(filename + '_segmented.jpg')


if __name__ == '__main__':

    print("called to do inference\n")
    makeInference("00dbd3c")
    # model = keras.models.load_model("model-save.h5", 
    #                             custom_objects={'bce_dice_loss': bce_dice_loss,
    #                                             'dice_coef': dice_coef,
    #                                            })

    # test_img_path = os.getcwd() + "/app/static/img/00dbd3c.jpg"
    # # test_img_path = "/app/static/img/00dbd3c.jpg"
    # imageAsArray = readImageAsArray(test_img_path)    
    # imageAsArrayCropped = cv2.resize(imageAsArray, (480, 320))   
    # imageAsArrayExpanded = np.expand_dims(imageAsArrayCropped, axis=0)
    # modelPrediction = model.predict(imageAsArrayExpanded)
    # predictionMasks = modelPrediction[0, ].round().astype(int)

    
    # maskedImage = copy.deepcopy(imageAsArrayCropped)
    # print(maskedImage.shape)

    # for row in range(320):
    #     for col in range(480):
    #         if(predictionMasks[row][col][0]==1):
    #             maskedImage[row][col] = [255,0,0]
    #         elif(predictionMasks[row][col][1]==1):
    #             maskedImage[row][col] = [0,255,0]
    #         elif(predictionMasks[row][col][2]==1):
    #             maskedImage[row][col] = [0,0,255]
    #         elif(predictionMasks[row][col][3]==1):
    #             maskedImage[row][col] = [100,100,100]
    #         else:
    #             maskedImage[row][col] = [0,0,0]

    
    # plt.imshow(imageAsArrayCropped, cmap='gray')
    # plt.imshow(maskedImage, cmap='gist_rainbow_r', alpha=0.5)
    # plt.savefig('segmented-cloud.jpeg')
