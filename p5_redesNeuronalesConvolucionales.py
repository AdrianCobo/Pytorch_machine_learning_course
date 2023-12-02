# https://pythonprogramming.net/convolutional-neural-networks-deep-learning-neural-network-pytorch/
# dataset: https://www.kaggle.com/datasets/tongpython/cat-and-dog/

# Convolution -> Create a sliding window that will compare features in data clusters.
# Pooling -> Generate a "new data" using a sliding window that will select the maximum value, the average, etc., of those features.
# pip install opencv-python numpy tqdm matplotlib

import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data

# data processing class
class DogsVSCats():
    IMG_SIZE = 50 # we are going to reshape the images to 50x50
    
    # directories of the data
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    TESTING = "PetImages/Testing"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []

    # dataset balance counter variables
    catcount = 0
    dogcount = 0


    # We want to iterate through these two directories, grab the images, resize, scale, convert the class to number (cats = 0, dogs = 1), 
    # and add them to our training_data.

    # All we're doing so far is iterating through the cats and dogs directories, and looking through all of the images 
    # and handle for the images:
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) # just makes one_hot matrix as targets. 
                        #example of np.eye(2)[1] -> [0. 1.]

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1
                    except Exception as e:
                        pass
                        #print(label, f, str(e))
        
        print(self.training_data[0])
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", np.array(self.training_data,dtype=object)) # dtype because [image,result] have differents sizes
        print('Cats:',dogsvcats.catcount)
        print('Dogs:',dogsvcats.dogcount)

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

# get the data
training_data = np.load("training_data.npy",allow_pickle=True)
print(len(training_data))

# split the data into X and y and convert it into a tensor

import torch

X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

import matplotlib.pyplot as plt

plt.imshow(X[0], cmap="gray")
plt.show()
print(y[0])