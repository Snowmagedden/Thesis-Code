#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import glob2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt


# In[23]:


#creating and cleaning the engagement data
engagement = pd.read_csv("all_engagement_data.csv")
engagement.drop([210,211,212,213,214,215],axis=0,inplace=True)
engagement['tag'] = engagement['ppnr'] + '_' + engagement['lesson'] + '_' + engagement['fragment']


# In[24]:


#creating the video file name dataframe
list_of_files = []
for filename in os.listdir('videos'):
    list_of_files.append(filename)
    
df = pd.DataFrame(list_of_files)
df.columns = ['video name']

#Creating a tag to be able to append with the engagement dataframe
df['tag'] = df['video name'].map(lambda x: x.strip('_front.avi').strip('_side.avi'))


# In[25]:


#merging the engagement and video file dataframes
result = pd.merge(df,
                 engagement[['tag', 'engagement_rating']],
                 on = 'tag') 


# In[7]:


#After having created image frames create a dataframe using the image frames
images = glob2.glob("image frames/*.jpg")
images_ = []
images_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    images_.append(images[i])
    
# storing the images and their class in a dataframe
frames = pd.DataFrame()
frames['image'] = images_

# converting the dataframe into csv file 
frames.to_csv('frames.csv',header=True, index=False, line_terminator='\n')


# In[26]:


# After cleaning the frames file using excel you should have three columns 'video name', 'frame', and 'path'
# for convience the completed frames file is also uploaded as a .csv
frames = pd.read_csv('frames.csv',sep = ';')


# In[27]:


#creating dataframe that has the video name, frame, path to frame and corresponding engagement rating
final = pd.merge(frames,
                       result[['video name', 'engagement_rating']],
                       on = 'video name') 


# In[2]:


#After manually seperating the videos into train and test 
split = pd.read_csv('75-25% split.csv', sep=';')


# In[29]:


#Merge the split csv file with the final_close dataframe
#This will result in a dataframe that is split in train and test in ascending order
final = pd.merge(split,
                 final[['video name','frame','path']],
                 on = 'video name')       # to keep the videos without engagement score add how = 'left' engagement score is NaN 
final.to_csv('final.csv',header=True, index=False, line_terminator='\n')

