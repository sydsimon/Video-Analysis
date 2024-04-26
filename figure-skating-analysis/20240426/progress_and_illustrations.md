Old approach: Round the video scores in the dataset to integers, use the obtained integers as classification labels to train the model, and the model predicts the category of the score.

New approach: Using the features of each frame of the video in the dataset and the temporal features of the video as independent variables, and the video score (decimal) as the dependent variable, the regression model is trained to score the video directly without going through the classification process. 

If there is time in the later stage, we will try more classification or regression models for comparative analysis.

Current results:

We randomly select 33 videos from each action classification in the dataset, evaluate the accuracy of the model's action classification and scoring, and write the results in 

"video. csv"

#This result is from the old model#

Action prediction accuracy: 97.98%
Average score difference (taking the average absolute value): 0.59
Maximum score difference: 3.18
Minimum score difference: 0.01