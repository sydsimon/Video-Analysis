import os
import json
import time
import re
import numpy as np
def parse_time_range(time_range_str):
    start_time, end_time = time_range_str.split(",")
    start_min, start_sec = start_time.split("-")
    end_min, end_sec = end_time.split("-")

    start_time_str = f"00:{int(start_min):02d}:{int(start_sec):02d}"
    end_time_str = f"00:{int(end_min):02d}:{int(end_sec):02d}"

    return start_time_str, end_time_str

def slice_3d_array(arr, start, end):
    return arr[int(start):int(end)]

def time_to_frame(time_range_str):
    start_time, end_time = time_range_str.split(",")
    start_min, start_sec = start_time.split("-")
    end_min, end_sec = end_time.split("-")
    start_time_frame = 25*(60*int(start_min) + int(start_sec))
    end_time_frame = 25*(60*int(end_min) + int(end_sec))
    return start_time_frame, end_time_frame

def video_cut_into_types(jsonfilepath,videofilepath):
    with open(jsonfilepath, 'r') as file:
        annotations = json.load(file)

    video_file = os.path.join(videofilepath)

    executed_element = annotations["executed_element"]

    values = executed_element.values()
    for i in values:
        element_folder_name = i["coarse_class"]
        element_folder_name = re.sub(r'[^\w\-_\. ]', '_', element_folder_name)
        element = i["element"]
        goe = "score_" + str(i["goe"])
        element_folder = os.path.join(element_folder_name)

        os.makedirs(element_folder, exist_ok=True)



        time_segments = i["time"]
        start_time, end_time = parse_time_range(time_segments)

        if os.path.isfile(video_file):

            output_file = os.path.join(element_folder, f"{goe}.mp4")

            count = 0
            while os.path.exists(output_file):
                output_file = os.path.join(element_folder, f"{goe}_{count}.mp4")
                count += 1

            os.system(f"ffmpeg -i {video_file} -ss {start_time} -to {end_time} -c copy {output_file}")
    return 0

def npz_cut_into_types(jsonfilepath,npzfilepath):
    with open(jsonfilepath, 'r') as file:
        annotations = json.load(file)
    npz_path = os.path.join(npzfilepath)
    npz_file = np.load(npzfilepath)
    data = npz_file["reconstruction"]

    executed_element = annotations["executed_element"]

    values = executed_element.values()
    for i in values:
        element_folder_name = i["element"]
        element_folder_name = re.sub(r'[^\w\-_\. ]', '_', element_folder_name)
        element_folder = os.path.join(element_folder_name)
        os.makedirs(element_folder, exist_ok=True)

        goe = i["goe"]

        time_segments = i["time"]
        start_frame, end_frame = time_to_frame(time_segments)

        newoutput = slice_3d_array(data,start_frame,end_frame)

        if os.path.isfile(npz_path):

            output_file = os.path.join(element_folder, f"{goe}")

            count = 1
            while os.path.exists(f"{output_file}.npz"):
                output_file = os.path.join(element_folder, f"{goe}_{count}")
                count += 1

            np.savez(output_file,newoutput)
    return 0

video_file_folder = r"D:\BaiduNetdiskDownload\FineFS\data\video\video"
json_file_folder = r"D:\BaiduNetdiskDownload\FineFS\data\annotation\annotation"
skeleton_file_folder = r"D:\BaiduNetdiskDownload\FineFS\data\skeleton\skeleton"

for i in range(850,1167,1):

    imput1 = f"{json_file_folder}/{i}.json"
    imput2 = f"{video_file_folder}/{i}.mp4"

    video_cut_into_types(imput1,imput2)
    print(i)

#for i in range(1166):
   # imput1 = f"{json_file_folder}/{i}.json"
  #  imput2 = f"{skeleton_file_folder}/{i}.npz"

 #   adasda = npz_cut_into_types(imput1,imput2)
#    print(i)
#
