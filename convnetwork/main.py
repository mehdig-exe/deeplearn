import os
import cv2
import numpy as np
from tqdm import tqdm

REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0 #counting to sort out any balance issues
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS: #iterating over the two directories
            print(label)
            for f in tqdm(os.listdir(label)): # iterating within the directories
            path = os.path.join(label, f) #joins the files with directory name
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # loads the file in grayscale
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) # resizes the files into 50x50
            self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

            if label == self.CATS:
                self.catcount += 1
            elif label == self.DOGS:
                self.dogcount += 1
