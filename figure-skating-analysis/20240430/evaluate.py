from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import csv
import pickle

with open('svr_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)




csv_file = open("videos_test_20040428.csv", mode='r', newline='')
csv_reader = csv.reader(csv_file)

# 跳过表头
next(csv_reader)

csv_files = open("videos_test_20240429.csv", mode='w', newline='')
csvs = csv.writer(csv_files)
csvs.writerow(
    ['Predicted_type', 'Predicted_type_probabilities', "score_-5", "pbblt", "score_-4", "pbblt", "score_-3", "pbblt",
     "score_-2", "pbblt", "score_-1", "pbblt", "score_0", "pbblt", "score_1", "pbblt", "score_2", "pbblt", "score_3",
     "pbblt", "score_4", "pbblt", "score_5", "pbblt", "action_score", 'predicted_score'])


for row in csv_reader:
    saves = []
    temp = []
    for i in range(24):
        if row[i] == 'jump':
            a = 100
        elif row[i] == 'sequence':
            a = 0
        elif row[i] == 'spin':
            a = -100
        else:
            a = float(row[i])
        temp.append(a)
    saves.append(temp)
    prediction_score = loaded_model.predict(saves)
    temp.append(row[24])
    temp.append(float(prediction_score[0]))
    print(temp)
    csvs.writerow(temp)

csv_file.close()