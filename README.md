# Thesis-Code
Using deep learning to predict total engagement assessment using videos of interactions between children and the NAO robot. 

The code in this repository closely follows and uses code from Harvey, M. (2019). Five video classification methods. GitHub repository:
https://github.com/harvitronix/five-video-classification-methods

# Abstract
The purpose of this study was the use of deep learning models to predict total engagement scores of video interaction between children and the NAO robot. Using video data this study explored the LSTM model with image sequences and Conv2D model with static images in predicting engagement. I found that the LSTM is a good deep learning model and outperforms the Conv2D model in predicting engagement on video data. As a result this study hoped to (1) assist educators in assessing the engagement level for the child with the NAO robot and determine if adjustments need to be made, and (2) alleviate researchers from the tedious and time-consuming task of annotating large number of videos.

## Conv2D
To prepare the data for the Conv2D model some excel and notepad needs to be used to clean the final dataframe and merge with results. In addition it is also used to manually split the train and test set.

### Conv2D model

![image](https://user-images.githubusercontent.com/66143690/93347247-240cdf80-f835-11ea-9a1f-d60f2db162ef.png)

### Results
![image](https://user-images.githubusercontent.com/66143690/93347656-9d0c3700-f835-11ea-967c-c44a59298673.png)

As expected the Conv2D model performed very poorly in using static images to predictal engagement score of a child in a video

## LSTM
For the LSTM model create three folders "train", "test" and "sequences". In addition the videos should be in the same folder as the .py files and not in a subfolder.

### LSTM model

![image](https://user-images.githubusercontent.com/66143690/93347563-849c1c80-f835-11ea-9b45-7581cbc8db51.png)

### Results

![image](https://user-images.githubusercontent.com/66143690/93347981-f96f5680-f835-11ea-9049-f3ea5a95b074.png)![image](https://user-images.githubusercontent.com/66143690/93348013-012efb00-f836-11ea-89dd-0d95610de009.png)

The LSTM model performs moderately well in predicting the engagement score of a child in a video
