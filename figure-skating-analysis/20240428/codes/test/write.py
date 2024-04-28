import csv
import random
import numpy as np
import tensorflow as tf
import os
from moviepy.editor import *

from test import predict_single_action,predict_score,predict_final_score
import re

from keras.models import load_model

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

IMAGE_HEIGHT , IMAGE_WIDTH = 60,60

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 25

# Specify the directory containing the UCF50 dataset.
DATASET_DIR = r"D:\BaiduNetdiskDownload\Fine_FS_3_categories"

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["spin","jump","sequence"]
CLASSES_LIST_for_sequence = ["score_-1","score_-2","score_0","score_1","score_2"]
CLASSES_LIST_for_spin = ["score_-1","score_2","score_0","score_1","score_-3"]
CLASSES_LIST_for_jump = ["score_-1","score_-2","score_-3","score_-4","score_-5","score_0","score_1","score_2","score_3","score_4","score_5"]

LRCN_model = load_model("LRCN_model_Classification_of_spin_jump_sequence.h5")
LRCN_model_for_spin = load_model("LRCN_model_Score_of_Spin_Accuracy_0.583427906036377.h5")
LRCN_model_for_jump = load_model("LRCN_model_Score_of_jump_Accuracy_0.2960088551044464.h5")
LRCN_model_for_sequence = load_model("LRCN_model_Score_of_Sequence_Accuracy_0.6449999809265137.h5")
# 指定文件夹路径
path = r"D:\BaiduNetdiskDownload\score_training"

# 定义要选择的子文件夹名称
folders = ["jump", "spin", "sequence"]

# 创建CSV文件并写入表头
csv_file = open("videos_test_20040428.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)


def get_score_number(score):
    return int(score.split('_')[-1])

# 对每个目标子文件夹进行处理
def write_csv_row(target_folders,folder_path,num,csv_write):
    for target_folder in target_folders:
        folder = os.path.join(folder_path, target_folder)
        # 检查子文件夹是否存在
        if not os.path.exists(folder):
            print(f"子文件夹 {target_folder} 不存在。")
            continue
        # 获取子文件夹下的所有视频文件
        video_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
        # 检查视频文件是否存在
        if len(video_files) == 0:
            print(f"子文件夹 {target_folder} 中没有找到视频文件。")
            continue
        # 随机选择100个视频
        random_videos = random.sample(video_files, k=min(num, len(video_files)))
        # 将所选择视频的文件夹名称和视频名称写入CSV文件
        for video in random_videos:
            input_video_file_path = rf"{folder_path}\{target_folder}\{os.path.basename(video)}"
            action_type = target_folder
            action_score = float(re.findall(r"-?\d+\.\d+", os.path.basename(video))[0])
            predicted_type,prob = predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
            predicted_score,labels_pbblt = predict_final_score(input_video_file_path, SEQUENCE_LENGTH, predicted_type, prob)
            score_difference = predicted_score - action_score
            #csv_writer.writerow([target_folder, os.path.basename(video),action_type,predicted_type,action_score,predicted_score,score_difference])
            #print(labels_pbblt)
            #print(action_score)
            scores = labels_pbblt[::2]  # 提取所有的score_n
            probabilities = labels_pbblt[1::2]  # 提取所有的概率
            sorted_array = [score for _, score in sorted(zip(scores, probabilities), key=lambda x: get_score_number(x[0]))]
            if predicted_type == "jump":
                final_score_pbblt = sorted_array
            elif predicted_type == "sequence":
                final_score_pbblt = [0,0,0] + sorted_array + [0,0,0]
            elif predicted_type == "spin":
                final_score_pbblt = [0,0] + [sorted_array[0]] + [0] + sorted_array[1:5] + [0,0,0]
            #print(final_score_pbblt)
            score_labels = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
            csv_file_row = [predicted_type,prob]
            for i in range(11):
                csv_file_row.append(score_labels[i])
                csv_file_row.append(final_score_pbblt[i])
            csv_file_row.append(action_score)
            print(csv_file_row)
            csv_write.writerow(csv_file_row)

# 关闭CSV文件
csv_writer.writerow(['Predicted_type','Predicted_type_probabilities',"score_-5","pbblt","score_-4","pbblt","score_-3","pbblt","score_-2","pbblt","score_-1","pbblt","score_0","pbblt","score_1","pbblt","score_2","pbblt","score_3","pbblt","score_4","pbblt","score_5","pbblt","action_score"])
write_csv_row(folders,path,100,csv_writer)
csv_file.close()