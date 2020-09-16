# Thesis-Code
Using deep learning to predict total engagement assessment using videos of interactions between children and the NAO robot. 

The code in this repository closely follows and uses code from Harvey, M. (2019). Five video classification methods. GitHub repository:
https://github.com/harvitronix/five-video-classification-methods

# Abstract
The purpose of this study was the use of deep learning models to predict total engagement scores of video interaction between children and the NAO robot. Using video data this study explored the LSTM model with image sequences and Conv2D model with static images in predicting engagement. I found that the LSTM is a good deep learning model and outperforms the Conv2D model in predicting engagement on video data. As a result this study hoped to (1) assist educators in assessing the engagement level for the child with the NAO robot and determine if adjustments need to be made, and (2) alleviate researchers from the tedious and time-consuming task of annotating large number of videos.

## Conv2D
To prepare the data for the Conv2D model some excel and notepad needs to be used to clean the final dataframe and merge with results. In addition it is also used to manually split the train and test set.

Example image of final dataframe

![image](https://user-images.githubusercontent.com/66143690/93349853-4a804a00-f838-11ea-913f-69661623ce9c.png)

### Conv2D model

![image](https://user-images.githubusercontent.com/66143690/93347247-240cdf80-f835-11ea-9a1f-d60f2db162ef.png)

### Results
![image](https://user-images.githubusercontent.com/66143690/93349230-85ce4900-f837-11ea-8c35-35b8adda27be.png)

As expected the Conv2D model performed very poorly in using static images to predictal engagement score of a child in a video

## LSTM
For the LSTM model create three folders "train", "test" and "sequences". In addition the videos should be in the same folder as the .py files and not in a subfolder.

### LSTM model

![image](https://user-images.githubusercontent.com/66143690/93347563-849c1c80-f835-11ea-9b45-7581cbc8db51.png)

### Results
LeakyRelu activation 

![image](https://user-images.githubusercontent.com/66143690/93349310-9c74a000-f837-11ea-9352-c9b276e3279d.png)![image](https://user-images.githubusercontent.com/66143690/93349456-cb8b1180-f837-11ea-9590-076d2f77b612.png)

Relu activation

![image](https://user-images.githubusercontent.com/66143690/93349021-430c7100-f837-11ea-93de-43e7531728e3.png)![image](https://user-images.githubusercontent.com/66143690/93349057-4b64ac00-f837-11ea-86a3-c2117f97404e.png)


The LSTM model performs moderately well in predicting the engagement score of a child in a video
