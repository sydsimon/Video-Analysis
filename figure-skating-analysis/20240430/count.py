import csv
import matplotlib.pyplot as plt
import numpy as np
# 打开CSV文件
csv_file = open("videos_test_3.csv", mode='r')
csv_reader = csv.reader(csv_file)

# 跳过表头
next(csv_reader)

# 初始化变量
total_records = 0
correct_predictions = 0
score_differences = []
action_scores = []
predicted_scores = []
# 逐行读取CSV文件
for row in csv_reader:
    action_type = row[2]
    predicted_type = row[3]
    action_score = row[4]
    predicted_score = row[5]
    action_scores.append(float(action_score))
    predicted_scores.append(float(predicted_score))
    score_difference = float(row[6])

    # 计算正确率
    if action_type == predicted_type:
        correct_predictions += 1
    total_records += 1

    # 记录score_difference
    score_differences.append(score_difference)

# 计算正确率
accuracy = correct_predictions / total_records

# 计算score_difference的平均值、最大值和最小值
average_difference = sum(score_differences) / len(score_differences)
max_difference = max(score_differences)
min_difference = min(score_differences)

# 输出结果
print("动作预测正确率: {:.2%}".format(accuracy))
print("平均差值: {:.2f}".format(average_difference))
print("最大差值: {:.2f}".format(max_difference))
print("最小差值: {:.2f}".format(min_difference))
x = action_scores
y = predicted_scores
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# 计算x和y的范围
x_range = x_max - x_min
y_range = y_max - y_min

# 规范化坐标轴范围，使得x和y轴的单位长度相同
# 这里我们假设希望x轴和y轴有相同的长度单位，即它们的范围相同
# 可以通过找到x和y范围中的较大者，并将其应用到另一个轴上来实现
max_range = max(x_range, y_range)

# 设置x和y轴的范围
plt.xlim([x_min - 0.05 * max_range, x_max + 0.05 * max_range])
plt.ylim([y_min - 0.05 * max_range, y_max + 0.05 * max_range])

# 绘制散点图
plt.scatter(x, y)
x2 = np.linspace(-10, 10, 1000)
y2 = x2
plt.plot(x2, y2)
# 添加标签和标题
plt.xlabel('action_scores')
plt.ylabel('predicted_scores')
plt.title('Score')

# 显示图表
plt.grid(True)  # 可选：添加网格线以便观察单位长度
plt.show()
