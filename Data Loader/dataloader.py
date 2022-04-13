# import standard PyTorch module
from torch.utils.data import Dataset
# importing image procressing tool 
from scipy import ndimage as ni
from skimage import io
# importing additional modules needed 
import numpy as np
import pandas as pd
import os

class ConfAndISM(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
      """
      ConfAndISM Inherits all of the useful modules and functions from the pre-built
      PyTorch class 'Dataset'. This class loads in the training images, normalises them,
      resizes the confocal images and shuffles them.

      Args:
      ---------
        -csv_file = The file containing the names of all of the Conf and ISM tif files.
        -root_dir = The directory in which the images and csv file are stored.
        -transform = Transforms the output to a user defined data type, for our 
                     purposes, tensors.
      """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
      """
      Returns the length of the object for which the method is called.

      Args:
      --------
        -self
      Returns:
      --------
        -len(self.annotations) = The length of the csv file
      """
        return len(self.annotations)

    def __getitem__(self, index):
      """
      Opens the training image files, at a specific index defined in the classmethod
      parameters. Converts the dtype for the image array to 'float32' so that more
      operations can be configured. For the Confocal images 'image' the array is zoomed 
      so that it goes from a (600,600) to (1200,1200) array. The image and labels are then 
      both normalized and transformed.

      Args:
      --------
        -index: The random index at which each image is, utilised in the Dataset class
                to shuffle the data.
      Returns:
      --------
        -image: The Confocal image tensor normalised and reshaped to be (1200,1200).
        -label: The ISM image tensor normalised.
      """
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = image.astype('float32')
        image = ni.zoom(image, [2, 2], order=1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = self.transform(image)

        label_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        label = io.imread(label_path)
        label = label.astype('float32')
        label = (label - np.min(label)) / (np.max(label) - np.min(label))
        label = self.transform(label)

        return image, label

