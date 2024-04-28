import cv2
import random
import numpy as np
import tensorflow as tf
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

def predict_single_action(video_file_path, SEQUENCE_LENGTH):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis=0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Release the VideoCapture object.
    video_reader.release()

    return predicted_class_name,predicted_labels_probabilities[predicted_label]


def predict_score(video_file_path, SEQUENCE_LENGTH, predicted_class_type):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    if predicted_class_type == "jump":
        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model_for_jump.predict(np.expand_dims(frames_list, axis=0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index.
        predicted_score = CLASSES_LIST_for_jump[predicted_label]

        # Display the predicted action along with the prediction confidence.
        print(f'Score Predicted: {predicted_score}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

        video_reader.release()

        return predicted_score

    if predicted_class_type == "spin":
        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model_for_spin.predict(np.expand_dims(frames_list, axis=0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index.
        predicted_score = CLASSES_LIST_for_spin[predicted_label]

        # Display the predicted action along with the prediction confidence.
        print(f'Score Predicted: {predicted_score}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

        video_reader.release()

        return predicted_score

    if predicted_class_type == "sequence":
        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model_for_sequence.predict(np.expand_dims(frames_list, axis=0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index.
        predicted_score = CLASSES_LIST_for_sequence[predicted_label]

        # Display the predicted action along with the prediction confidence.
        print(f'Score Predicted: {predicted_score}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

        video_reader.release()

        return predicted_score



def predict_final_score(video_file_path, SEQUENCE_LENGTH, predicted_class_type, pbblt):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    count1 = 1
    returns = []
    if predicted_class_type == "jump":
        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model_for_jump.predict(np.expand_dims(frames_list, axis=0))



        for i in range(len(predicted_labels_probabilities)):
            class_probabilities = predicted_labels_probabilities[i]
            for j in range(len(class_probabilities)):
                class_name = CLASSES_LIST_for_jump[j]
                probability = class_probabilities[j]
                print(f"预测结果：{class_name}，概率：{probability}")
                returns.append(class_name)
                returns.append(probability)
                count1 = count1 + float(re.findall("-?\d",class_name)[0])*float(probability)
        count1 = count1 - 1/pbblt
        print(count1)
        video_reader.release()

        return count1,returns

    if predicted_class_type == "spin":
        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = LRCN_model_for_spin.predict(np.expand_dims(frames_list, axis=0))


        for i in range(len(predicted_labels_probabilities)):
            class_probabilities = predicted_labels_probabilities[i]
            for j in range(len(class_probabilities)):
                class_name = CLASSES_LIST_for_spin[j]
                probability = class_probabilities[j]
                print(f"预测结果：{class_name}，概率：{probability}")
                returns.append(class_name)
                returns.append(probability)
                count1 = count1 + float(re.findall("-?\d", class_name)[0]) * float(probability)
        count1 = count1 - 1 / pbblt
        print(count1)
        video_reader.release()

        return count1,returns

    if predicted_class_type == "sequence":

        predicted_labels_probabilities = LRCN_model_for_sequence.predict(np.expand_dims(frames_list, axis=0))


        for i in range(len(predicted_labels_probabilities)):
            class_probabilities = predicted_labels_probabilities[i]
            for j in range(len(class_probabilities)):
                class_name = CLASSES_LIST_for_sequence[j]
                probability = class_probabilities[j]
                print(f"预测结果：{class_name}，概率：{probability}")
                returns.append(class_name)
                returns.append(probability)
                count1 = count1 + float(re.findall("-?\d", class_name)[0]) * float(probability)
        count1 = count1 - 1 / pbblt
        print(count1)
        video_reader.release()

        return count1,returns



# Perform Single Prediction on the Test Video.
#input_video_file_path = input("")
#type1,prob = predict_single_action(input_video_file_path, SEQUENCE_LENGTH)
#score,label_pbblt = predict_final_score(input_video_file_path, SEQUENCE_LENGTH, type1, prob)

