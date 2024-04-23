A scoring system for figure skating training based on visual recognition 

Progress: 
we finished analyzing the dataset by cutting and renaming the videos of figure skating competitions into different folders with labels movement type and score.

So far, we have started training the model using LRCN. We have classified the movements into three basic categories: jump, spin and step sequence.
We also have trained the three categories separately to predict the scores for each movement using the models but the accuracy is still low. 
We have score -5 to 5 in the category “jump”, the accuracy for validation is 0.296.
We have only score -2 to 2 in the category “sequence” in our dataset, so we used it to train the model. The validation accuracy is 0.645.
We have only score -3 to 1 in the category “spin” in our dataset, so we used it to train the model. The validation accuracy is 0.583.
We have codes for direct predicion based on this old model.



Experiments:

Accuracy is low on testing.

Remaining implementation plan:

We will finish finding a different approach to achieve higher accuracy and training a new regression model for each of the three predicted action types to achieve scoring (and classification of movements if we have time) by this month. 

How duties for each group member are organized:

Terry: Training the model and doing most of the coding.
Eason: Mostly coding with Terry and finding the dataset. Learn and teaching the rest of the group how the models and the networks work.
Alice: Finding the data, doing most of the paper work and providing basic information needed.

Our data is collected from GitHub and it is a data set called “Fine_FS” the data set includes the videos of hundreds of programs from 2017 to 2018, and they were split by different movements and stored from the highest score to the lowest for each movement.

Run "videocut.py" under the folders in the dataset to cut videos into shapes which are ready for training.

We have all the codes of our old approach under \codes\old-approach 

We have a failed code of our new approach. we are trying to solve the problem now.
