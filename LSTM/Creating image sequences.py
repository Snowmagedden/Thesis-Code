#!/usr/bin/env python
# coding: utf-8


import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import os.path
from tqdm import tqdm
import csv
import glob
import pickle


# @Harvey
class DataSet():

    def __init__(self, seq_length=120, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('sequences')
        self.max_frames = 125  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()
        

    def get_data(self):
        """Load our data from file."""
        with open(os.path.join('data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(filter(None,reader))
        return data

                
    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes


# @Harvey
def get_frames_for_sample(sample):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    path = os.path.join(sample[0]+"//", sample[1]+"//")
    filename = sample[2]
    path_image = (os.path.join(path, filename))
    images = sorted(glob.glob(os.path.join(path, filename +'*jpg')))
    return images


# @Harvey
class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = ResNet50(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )


    def extract(self, image_path):
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224,3))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features


# @Harvey
# Set defaults.
seq_length = 120
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()   

# Loop through data.
lst = []
label_lst = []
train_test_list = []
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = os.path.join('sequences'+"//", video[2] + '-' + str(seq_length) +         '-features')  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        print('There is already a sequence of file:',video[2])
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = get_frames_for_sample(video)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence.
    np.save(path, sequence)
    lst.append(sequence)
    label_lst.append(video[1])
    train_test_list.append(video[0])
    pbar.update(1)

pbar.close()


with open("lst_120.txt", "wb") as fp:   #Pickling
    pickle.dump(lst, fp)
with open("label_lst_120.txt", "wb") as fp:   #Pickling
    pickle.dump(label_lst, fp)

