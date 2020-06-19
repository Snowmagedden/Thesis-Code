#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from tqdm import tqdm
import cv2


#Creating a list of videos
list_of_files = []
for filename in os.listdir('videos'):
    list_of_files.append(filename)

    
#Creating a dataframe out of the list
df = pd.DataFrame(list_of_files)
df.columns = ['video name']


# @Kang&Atul
for i in tqdm(range(df.shape[0])):
    count = 0
    videoFile = df['video name'][i]
    # Opens the Video file
    cap= cv2.VideoCapture('videos/'+ videoFile) #videos folder
    i=0   # setting this to 0 gives first frame but not last// setting to 1 gives last frame but not first
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%125 == 0:      # total of 3000 frames per image   (i = 3000 %125 is divided by 125 so this gives 24 frames)
            filename ='image frames/'+ videoFile + "_frame%d.jpg" % count;count+=1  #saves into image frames folder
            cv2.imwrite(filename, frame)
        i+=1
 
    cap.release()
    cv2.destroyAllWindows()

