from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import csv
import pickle


with open("videos_test_20040428 - 副本.csv",'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
# 跳过表头


    X = []
    y = []
    for row in csv_reader:
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
        X.append(temp)
        y.append(float(row[24]))




    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 创建SVR模型并进行训练
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svr.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f"均方误差：{mse}")

with open('svr_model.pkl', 'wb') as f:
    pickle.dump(svr, f)
print(svr)