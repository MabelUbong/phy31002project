import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import os
import random

# changes tiff images into arrays
def loadtiffs(img):

      imgArray = np.zeros((img.size[1], img.size[0], img.n_frames), np.uint16)
      for I in range(img.n_frames):
        img.seek(I)
        imgArray[:, :, I] = np.asarray(img)
      img.close()
      return(imgArray)

# loads images from directory in google drive
def Directory(dname,n):
    img_array = []

    for fname in os.listdir(dname):
        im = Image.open(os.path.join(dname, fname)) # finds all the tiff files in the specific directory
        img_array.append(im)

#removes unwanted data
    if dname == '/content/drive/Shareddrives/Team Net/Training Data/Conf_Train':
        img_array.pop(19)
        img_array.pop(19)
        img_array.pop(19)
        img_array.pop(19)
        img_array.pop(30)
        img_array.pop(30)
        img_array.pop(30)

# Now to randomise the array
# Create an list from 1 to n
    indexes = []
    for i in range(0, len(img_array)):
        indexes.append(i)

#create a random list from the indexes
    rand_indexes = random.sample(indexes, len(indexes))

#randomising images using the random indexes
    img_rand_array = []
