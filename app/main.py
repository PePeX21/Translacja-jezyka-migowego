### IMPORTS ###
import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from pydantic import BaseModel
from tempfile import NamedTemporaryFile

import time
from numpy import zeros
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------------------------------------




### API PART ###
class Item(BaseModel):
    link: str

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)

# --------------------------------------------------------------------------------------------------------














### FUNCTIONS WORKING WITH MP HOLISTIC MODEL ###

# Initialazing mp holistic for keypoints
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


# Finding landmarks with model holistic
def mediapipe_detection(image, model):
    # creating contrast on img
    '''
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # applying CLAHE to L-channel   - feel free to try different values for the limit and grid size
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b)) # merge the CLAHE enhanced L-channel with the a and b channel
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) #onverting image from LAB Color model to BGR color spcae
    '''

    # detections
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color conversion BGR 2 RGB
    image.flags.writeable = False  # image is no longer writeable
    results = model.process(image)  # make prediction
    image.flags.writeable = True  # image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # color conversion RGB 2 BGR
    return image, results




# Drawing landmarks
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )  # draw pose connections

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )  # draw left hand connections

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )  # draw right hand connections




# Preparing vector of date
def extract_keypoints(results):
    # fitting resolution different bettwen mobile phone(1280, 720) and computer (480, 640)
    # - data set was created with computer with around 3 times less pixels per square
    x_resolution_difference = 0.75 # 3/4
    y_resolution_difference = 1.778 # 16/9

    if (results.pose_landmarks == None):  # if there is no landmarks
        return [], [], [], [], [],

    else:
        # face
        x = 0
        y = 0
        for i in range(11):
            if results.pose_landmarks.landmark[i].visibility > 0.95:
                x += results.pose_landmarks.landmark[i].x * x_resolution_difference
                y += results.pose_landmarks.landmark[i].y * y_resolution_difference
            else:
                return [], [], [], [], [],
        face = np.array([x / 11, y / 11])  # the middle point from all points on face

        # pose
        pose_left = []
        pose_right = []

        if (results.pose_landmarks.landmark[11].visibility > 0.80 and results.pose_landmarks.landmark[
            12].visibility > 0.80):  # middle point between choulders
            pose_left.append(
                ((results.pose_landmarks.landmark[11].x * x_resolution_difference + results.pose_landmarks.landmark[12].x * x_resolution_difference) / 2) - face[0])
            pose_left.append(
                ((results.pose_landmarks.landmark[11].y * y_resolution_difference + results.pose_landmarks.landmark[12].y * y_resolution_difference) / 2) - face[1])
            pose_right.append(
                ((results.pose_landmarks.landmark[11].x * x_resolution_difference + results.pose_landmarks.landmark[12].x * x_resolution_difference) / 2) - face[0])
            pose_right.append(
                ((results.pose_landmarks.landmark[11].y * y_resolution_difference + results.pose_landmarks.landmark[12].y * y_resolution_difference) / 2) - face[1])
        else:
            pose_left.append(0)
            pose_left.append(0)
            pose_right.append(0)
            pose_right.append(0)

        for i in range(11, 17):  # points from pose - arms and shoulder
            if (results.pose_landmarks.landmark[i].visibility > 0.80):
                if (i % 2 == 0):
                    pose_right.append(-1 * (results.pose_landmarks.landmark[i].x * x_resolution_difference - face[0]))
                    pose_right.append(results.pose_landmarks.landmark[i].y * y_resolution_difference - face[1])
                else:
                    pose_left.append(results.pose_landmarks.landmark[i].x * x_resolution_difference - face[0])
                    pose_left.append(results.pose_landmarks.landmark[i].y * y_resolution_difference - face[1])
            else:
                if (i % 2 == 0):
                    pose_right.append(0)
                    pose_right.append(0)
                else:
                    pose_left.append(0)
                    pose_left.append(0)

        pose_left = np.array(pose_left)
        pose_right = np.array(pose_right)

        # right hand
        if (results.right_hand_landmarks == None):
            right_hand = []
        else:
            right_hand = np.array(
                [[-1 * (res.x * x_resolution_difference - face[0]), res.y * y_resolution_difference - face[1]] for res in results.right_hand_landmarks.landmark]).flatten()

        # left hand
        if (results.left_hand_landmarks == None):
            left_hand = []
        else:
            left_hand = np.array(
                [[res.x * x_resolution_difference - face[0], res.y * y_resolution_difference - face[1]] for res in results.left_hand_landmarks.landmark]).flatten()

        return face, pose_left, pose_right, left_hand, right_hand

# --------------------------------------------------------------------------------------------------------














### FUNCTIONS FOR extract_sequence ###
def init_global_variables_on_zero(all_variables=True):
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand

    number_of_blank_pose_left = 0
    number_of_blank_pose_right = 0
    number_of_blank_left_hand = 0
    number_of_blank_right_hand = 0

    vector_of_blank_pose_left = []
    vector_of_blank_pose_right = []
    vector_of_blank_left_hand = []
    vector_of_blank_right_hand = []

    if all_variables:
        all_frames_pose_left = []
        all_frames_pose_right = []
        all_frames_left_hand = []
        all_frames_right_hand = []

        frame_num = 1
        all_keypoints = []
        left_zeros_fixed = False
        right_zeros_fixed = False




def counting_number_of_blanks():
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    # checking existence of landmarks on body and hands
    if len(left_hand) == 0:
        left_hand = np.zeros(42)
        vector_of_blank_left_hand.append(1)
    else:
        vector_of_blank_left_hand.append(0)

    if len(right_hand) == 0:
        right_hand = np.zeros(42)
        vector_of_blank_right_hand.append(1)
    else:
        vector_of_blank_right_hand.append(0)

    if len(pose_left) - np.count_nonzero(pose_left) != 0:
        vector_of_blank_pose_left.append(1)
    else:
        vector_of_blank_pose_left.append(0)

    if len(pose_right) - np.count_nonzero(pose_right) != 0:
        vector_of_blank_pose_right.append(1)
    else:
        vector_of_blank_pose_right.append(0)

    # number of blanks in last 12 frames not including last one
    if (frame_num > sequence_length):
        vector_of_blank_pose_left = vector_of_blank_pose_left[-31:]
        vector_of_blank_pose_right = vector_of_blank_pose_right[-31:]
        vector_of_blank_left_hand = vector_of_blank_left_hand[-31:]
        vector_of_blank_right_hand = vector_of_blank_right_hand[-31:]

    number_of_blank_pose_left = np.count_nonzero(vector_of_blank_pose_left[-31:-1])
    number_of_blank_pose_right = np.count_nonzero(vector_of_blank_pose_right[-31:-1])
    number_of_blank_left_hand = np.count_nonzero(vector_of_blank_left_hand[-31:-1])
    number_of_blank_right_hand = np.count_nonzero(vector_of_blank_right_hand[-31:-1])




# Updating hole story for single part
def update_all_frames():
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    all_frames_pose_left.append(pose_left)
    all_frames_pose_right.append(pose_right)
    all_frames_left_hand.append(left_hand)
    all_frames_right_hand.append(right_hand)

    # extracting last 13 frames
    if (frame_num > sequence_length):
        all_frames_pose_left = all_frames_pose_left[-31:]
        all_frames_pose_right = all_frames_pose_right[-31:]
        all_frames_left_hand = all_frames_left_hand[-31:]
        all_frames_right_hand = all_frames_right_hand[-31:]




# counting variation
def fixing_zeros_and_counting_varations(all_frames_part, check_varation=True, fix_zeros=True):
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    varations = []
    drop_varation = []
    col = []
    # creating vector of colmun
    for it in range(len(all_frames_part[0])):
        for frame in all_frames_part:
            col.append(frame[it])

        # fixing zeros in whole 12 frame sequance
        if fix_zeros:
            for it2 in range(1, len(col) - 1):
                if (col[it2] == 0 and col[it2 + 1] == 0):
                    col[it2] = col[it2 - 1]
                    all_frames_part[it2][it] = col[it2 - 1]
                if (col[it2] == 0):
                    col[it2] = (col[it2 - 1] + col[it2 + 1]) / 2
                    all_frames_part[it2][it] = col[it2]
        # fixing zeros only on the end of sequence
        else:
            if (col[29] == 0 and col[30] == 0):
                col[29] = col[28]
                all_frames_part[29][it] = col[28]
            if (col[29] == 0):
                col[29] = (col[28] + col[30]) / 2
                all_frames_part[29][it] = col[29]

        # variation from every colmun
        if check_varation:
            # counting varation
            col = 10 * np.array(col[:30])  # *10 working good for next if with treshold
            srednio = sum(col) / sequence_length
            varations.append(np.sum((np.array(col) - srednio) ** 2) / sequence_length)

            drop_srednio = sum(col[-10:]) / 10
            drop_varation.append(np.sum((np.array(col[-10:]) - drop_srednio) ** 2) / 10)
        col = []

    print(np.sum(drop_varation), np.sum(varations))
    # return if check varation
    if check_varation:
        if (np.sum(varations) >= 2):
            if np.sum(drop_varation) < 0.4:
                if np.sum(drop_varation) < 0.1:  # super drop
                    print("super DROP")
                    return [1, 1, True, True]
                else:
                    return [1, 1, True, False]

                #return [1, 1, True, False]
            else:
                return [1, 1, False, False]
        elif (np.sum(varations) < 0.6):
            #if np.sum(drop_varation) < 0.4:
            #    return [1, 0, True, False]
            #else:
            #    return [1, 0, False, False]
            return [1, 0, False, False]
        else:
            if np.sum(drop_varation) < 0.4:
                return [1, -404, True, False]
            else:
                return [1, -404, False, False]

    # return for fixiing and not fixing zeros
    return [1, 404]

# --------------------------------------------------------------------------------------------------------














### LOGIC ###
def extract_sequence(results):
    global frame_num, all_keypoints, left_zeros_fixed, right_zeros_fixed
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    # extracting keypoints
    face, pose_left, pose_right, left_hand, right_hand = extract_keypoints(results)

    if len(face) == 0:  # if there is no landmarks
        print("there is nothing on sreen")
        init_global_variables_on_zero()
        return 404, False


    # checking how many blanks is in every single part of body in last 12 frames
    counting_number_of_blanks()

    # updating whole story for single part
    update_all_frames()

    if (frame_num < sequence_length + 1):
        frame_num += 1
        return 404, False

    else:

        lh = [404, 404, False, False]
        lp = [404, 404, False, False]
        ph = [404, 404, False, False]
        pp = [404, 404, False, False]
        second_chance_for_2_hands = False


        # left hand and arm cant have more then 2 blanks frame and first analizng frame cant contain only zeros
        if (number_of_blank_left_hand <= 2 and number_of_blank_pose_left <= 2 and all_frames_left_hand[0][
            0] != 0 and len(pose_left) - np.count_nonzero(all_frames_pose_left[0]) == 0):
            if (left_zeros_fixed):
                lh = fixing_zeros_and_counting_varations(all_frames_left_hand, True, False)
                lp = fixing_zeros_and_counting_varations(all_frames_pose_left, False, False)
            else:
                lh = fixing_zeros_and_counting_varations(all_frames_left_hand)
                lp = fixing_zeros_and_counting_varations(all_frames_pose_left, False)
        # second chance for 2 hands, bigger tollerant for blanks frames, mediapipe is not perfect
        elif (number_of_blank_left_hand <= 5 and number_of_blank_pose_left <= 5 and all_frames_left_hand[0][
            0] != 0 and len(pose_left) - np.count_nonzero(all_frames_pose_left[0]) == 0):
            second_chance_for_2_hands = True

            if (left_zeros_fixed):
                lh = fixing_zeros_and_counting_varations(all_frames_left_hand, True, False)
                lp = fixing_zeros_and_counting_varations(all_frames_pose_left, False, False)
            else:
                lh = fixing_zeros_and_counting_varations(all_frames_left_hand)
                lp = fixing_zeros_and_counting_varations(all_frames_pose_left, False)

        # right hand and arm cant have more then 2 blanks frame and first analizng frame cant contain only zeros
        if (number_of_blank_right_hand <= 2 and number_of_blank_pose_right <= 2 and all_frames_right_hand[0][
            0] != 0 and len(pose_right) - np.count_nonzero(all_frames_pose_right[0]) == 0):
            if (right_zeros_fixed):
                ph = fixing_zeros_and_counting_varations(all_frames_right_hand, True, False)
                pp = fixing_zeros_and_counting_varations(all_frames_pose_right, False, False)
            else:
                ph = fixing_zeros_and_counting_varations(all_frames_right_hand)
                pp = fixing_zeros_and_counting_varations(all_frames_pose_right, False)

        # second chance for 2 hands, bigger tollerant for blanks frames, mediapipe is not perfect
        elif (number_of_blank_right_hand <= 5 and number_of_blank_pose_right <= 5 and all_frames_right_hand[0][
            0] != 0 and len(pose_right) - np.count_nonzero(all_frames_pose_right[0]) == 0 and not ((lh[1] == 1 and not lh[3]) or (ph[1] == 1 and not ph[3]))):
            second_chance_for_2_hands = True

            if (right_zeros_fixed):
                ph = fixing_zeros_and_counting_varations(all_frames_right_hand, True, False)
                pp = fixing_zeros_and_counting_varations(all_frames_pose_right, False, False)
            else:
                ph = fixing_zeros_and_counting_varations(all_frames_right_hand)
                pp = fixing_zeros_and_counting_varations(all_frames_pose_right, False)


        # not good data for prediction: 2 blanks next to eachother or too much blanks frame
        if (lh[0] == 404 or lp[0] == 404):
            left_zeros_fixed = False
        else:
            left_zeros_fixed = True
        if (ph[0] == 404 or pp[0] == 404):
            right_zeros_fixed = False
        else:
            right_zeros_fixed = True
        if (left_zeros_fixed == False and right_zeros_fixed == False):
            print("coutch not enough landmarks ")
            return 404, False


        # 2 hands are moving
        elif (lh[1] == 1 and not lh[3] and ph[1] == 1 and not ph[3]):
            print(
                "do sieci na 2 rece")  # TO DO ... jak bd sie jebac moge polaczyc pose'y z soba i wyjebac jeden wspolny pnkt#####

            for i in range(sequence_length):
                keypoints = np.concatenate([all_frames_pose_left[i], all_frames_left_hand[i], all_frames_pose_right[i],
                                            all_frames_right_hand[i]])
                all_keypoints.append(keypoints)

            # drop variantion
            if(lh[2] or ph[2]):
                return 2, True
            else:
                return 2, False

        # stopping if there was second chance for 2 hands
        elif (second_chance_for_2_hands == True):
            print("from second chance for 2 hands")
            # drop variantion
            if(lh[2] or ph[2]):
                return 404, True
            else:
                return 404, False
            #return 404, False

        # 1 hand is moving
        elif ((lh[1] == 1 and not lh[3]) or (ph[1] == 1 and not ph[3])):
            print("do sieci dla 1 reki")
            if (lh[1] == 1 and not lh[3]):
                for i in range(sequence_length):
                    keypoints = np.concatenate([all_frames_pose_left[i], all_frames_left_hand[i]])
                    all_keypoints.append(keypoints)

                # drop variantion
                if (lh[2] or (lh[1] == 1  and ph[1] == 1)): # 2 czlon super drop spadl z gory
                    return 1, True
                else:
                    return 1, False
            elif(ph[1] == 1 and not ph[3]):
                for i in range(sequence_length):
                    keypoints = np.concatenate([all_frames_pose_right[i], all_frames_right_hand[i]])
                    all_keypoints.append(keypoints)

                # drop variantion
                if (ph[2] or (lh[1] == 1  and ph[1] == 1)): # 2 czlon super drop spadl z gory
                    return 1, True
                else:
                    return 1, False

        # no move detected
        elif (lh[1] != -404 and ph[1] != -404 and lh[1] == 0 or ph[1] == 0):
            print("do sieci dla nie ruchomych rak")
            # jak nie chce choke sie jebalo to powybierac wszystkie kombinacje i dla przypadku kiedy jest tylko jedna reka
            # to druga wypelnic zerami
            all_keypoints = np.concatenate(
                [sum(all_frames_pose_left) / sequence_length, sum(all_frames_left_hand) / sequence_length,
                 sum(all_frames_pose_right) / sequence_length, sum(all_frames_right_hand) / sequence_length])

            if (lh[1] == 1  or ph[1] == 1): # super drop spadl z gory
                return 0, True
            else:
                return 0, False
        else:
            print("ni to w ruchu anie nie w ruchu")
            # drop variantion - moze byc uproszczone miedzy 1 i 2 rekoma
            if (lh[2] or ph[2]):
                return 404, True
            else:
                return 404, False

# --------------------------------------------------------------------------------------------------------














### FOLDER INFO ###
no_sequences = 50
sequence_length = 30
DATA_PATH = os.path.join('MP_Data')

# bez rak
actions0 = np.array(['choke', 'eat', 'hurt', 'I/me'])
colors0 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245)]

#1reka
actions1 = np.array(['arm', 'call', 'dizzy', 'drink', 'fall down', 'hello', 'infection', 'need', 'please', 'thirsty'])
colors1 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (255, 153, 51) ,(245,117,16), (117,245,16)]

#2reka
actions2 = np.array(['alergy', 'hit', 'hospital', 'in', 'thanks', 'want'])
colors2 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16)]

# importing model
model0 = keras.models.load_model("sieci/modelOneFrame_200.h5")
model1 = keras.models.load_model("sieci/model_biderictional_pure_1hand.h5")
model2 = keras.models.load_model("sieci/model_biderictional_pure_2hand.h5")

colors0 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245)]
colors1 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (255, 153, 51) ,(245,117,16), (117,245,16)]
colors2 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16)]

# --------------------------------------------------------------------------------------------------------














### MAiN LOOP ###
def processVideo(video):
    global frame_num, all_keypoints
    global number_of_blank_pose_left, number_of_blank_pose_right, number_of_blank_left_hand, number_of_blank_right_hand
    global vector_of_blank_pose_left, vector_of_blank_pose_right, vector_of_blank_left_hand, vector_of_blank_right_hand
    global all_frames_pose_left, all_frames_pose_right, all_frames_left_hand, all_frames_right_hand
    global pose_left, pose_right, left_hand, right_hand

    init_global_variables_on_zero()
    sentence = []
    sentence.append("no_landmarks")
    clean = True
    last_one = ""
    last_one_counter = -1
    threshold = 0.7

    last_nentwork = -1
    stop = 0
    stop_still_max = False

    counter_0 = 0
    counter_1 = 0
    counter_2 = 0

    xd_counter = 0
    xd_drop = False

    top_bar_color = (245, 117, 16)



    cv2.namedWindow("OpenCV ladnmark_test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("OpenCV ladnmark_test",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)


    cap = cv2.VideoCapture(video)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        while cap.isOpened():
            # read feed
            ret, frame = cap.read()
            if not ret:
                print("EOF")
                break

            # Use Flip code 0 to flip vertically
            frame = cv2.flip(frame, -1)

            # make detections
            image, results = mediapipe_detection(frame, holistic)

            # draw landmarks
            draw_styled_landmarks(image, results)

            # extract sequence
            network_choose, xd = extract_sequence(results)

            if(xd):
                xd_counter +=1
            else:
                xd_counter = 0

            if(xd_counter >= 10 and (last_nentwork == 2 or last_nentwork == 1)):
                xd_drop = True

            if (stop != 0):
                stop -= 1
                if (stop > 3):
                    colors0 = [(0, 0, 255) ,(0, 0, 255), (0, 0, 255) ,(0, 0, 255)]
                    colors1 = [(0, 0, 255) ,(0, 0, 255), (0, 0, 255) ,(0, 0, 255), (0, 0, 255) ,(0, 0, 255),  (0, 0, 255) ,(0, 0, 255), (0, 0, 255), (0, 0, 255)]
                    colors2 = [(0, 0, 255) ,(0, 0, 255), (0, 0, 255) ,(0, 0, 255), (0, 0, 255) ,(0, 0, 255)]
                    top_bar_color = (0, 0, 255)
                else:
                    colors0 = [(255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)]
                    colors1 = [(255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)]
                    colors2 = [(255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255)]
                    top_bar_color = (255, 0, 255)
            else:
                colors0 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245)]
                colors1 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16), (16,117,245), (255, 153, 51) ,(245,117,16), (117,245,16)]
                colors2 = [(255, 153, 51) ,(245,117,16), (117,245,16), (16,117,245), (245,117,16), (117,245,16)]
                top_bar_color = (245, 117, 16)
            print("stop:", stop, " xd: ", xd, " xd_counter:", xd_counter, " xd_drop:q", xd_drop)
            if(xd_drop):
                top_bar_color = (0, 255, 0)




            # choosing type of network
            if network_choose == 0 and (stop == 0 or (last_nentwork == network_choose and not xd_drop) or (xd_drop and last_nentwork != network_choose)):
                counter_0 += 1
                if(counter_0 > 5):
                    counter_1 = 0
                    counter_2 = 0
                    xd_drop = False

                # making prediction
                res = model0.predict(np.expand_dims(all_keypoints, axis=0))[0]
                # blokada po detekcji :)
                if actions0[np.argmax(res)] == sentence[-1] and stop_still_max:
                    if stop < 16:
                        stop = 16
                else:
                    stop_still_max = False

                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:

                    # last 5 frames must predict the same sign
                    if (last_one == actions0[np.argmax(res)]):
                        last_one_counter += 1
                    else:
                        last_one = actions0[np.argmax(res)]
                        last_one_counter = 1
                        stop_still_max = False

                    # blokada po detekcji :)
                    if actions0[np.argmax(res)] == sentence[-1] and stop_still_max:
                        stop = 20
                    else:
                        stop_still_max = False

                    if (last_one_counter > 4 and (counter_0 > 15 or xd_drop) or (clean and last_one_counter > 1)):

                        # no_landmarks deleting
                        if (clean):
                            sentence = []
                            clean = False

                        if len(sentence) > 0:

                            if actions0[np.argmax(res)] != sentence[-1] and (stop == 0 or xd_drop):
                                print("Dodano znak, counter:", counter_0)
                                if(stop != 0 and xd_drop):
                                    counter_1 = 0
                                    counter_2 = 0
                                    sentence = sentence[:-1]
                                if (last_nentwork != network_choose):
                                    xd_counter = 0
                                    xd_drop = False


                                sentence.append(actions0[np.argmax(res)])
                                stop = 20
                                stop_still_max = True
                        else:
                            sentence.append(actions0[np.argmax(res)])
                            stop = 20
                            stop_still_max = True

                        last_nentwork = 0

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # showing probabilities for every word
                for num, prob in enumerate(res):
                    cv2.rectangle(image, (0, 60 + num * 30), (int(prob * 100), 90 + num * 30), colors0[num], -1)
                    cv2.putText(image, actions0[num], (0, 85 + num * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)




            elif network_choose == 1 and (stop == 0 or (last_nentwork == network_choose and not xd_drop) or (xd_drop and last_nentwork != network_choose)):
                counter_1 += 1
                if (counter_1 > 5):
                    counter_0 = 0
                    counter_2 = 0
                    xd_drop = False

                # making prediction
                res = model1.predict(np.expand_dims(all_keypoints, axis=0))[0]
                # blokada po detekcji :)
                if actions1[np.argmax(res)] == sentence[-1] and stop_still_max:
                    if stop < 16:
                        stop = 16
                else:
                    stop_still_max = False

                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:

                    # last 5 frames must predict the same sign
                    if (last_one == actions1[np.argmax(res)]):
                        last_one_counter += 1
                    else:
                        last_one = actions1[np.argmax(res)]
                        last_one_counter = 1
                        stop_still_max = False

                    # blokada po detekcji :)
                    if actions1[np.argmax(res)] == sentence[-1] and stop_still_max:
                        stop = 20
                    else:
                        stop_still_max = False

                    if (last_one_counter > 4 and (counter_1 > 15 or xd_drop) or (clean and last_one_counter > 1)):
                        # no_landmarks deleting
                        if (clean):
                            sentence = []
                            clean = False

                        if len(sentence) > 0:
                            if actions1[np.argmax(res)] != sentence[-1] and (stop == 0 or xd_drop):
                                print("Dodano znak, counter:", counter_1)
                                if (stop != 0 and xd_drop):
                                    counter_0 = 0
                                    counter_2 = 0
                                    sentence = sentence[:-1]
                                if (last_nentwork != network_choose):
                                    xd_counter = 0
                                    xd_drop = False

                                sentence.append(actions1[np.argmax(res)])
                                stop = 20
                                stop_still_max = True
                        else:
                            sentence.append(actions1[np.argmax(res)])
                            stop = 20
                            stop_still_max = True
                        last_nentwork = 1

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # showing probabilities for every word
                for num, prob in enumerate(res):
                    cv2.rectangle(image, (0, 60 + num * 30), (int(prob * 100), 90 + num * 30), colors1[num], -1)
                    cv2.putText(image, actions1[num], (0, 85 + num * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)




            elif network_choose == 2 and (stop == 0 or last_nentwork == network_choose):
                counter_2 += 1
                if (counter_2 > 5):
                    counter_0 = 0
                    counter_1 = 0

                # making prediction
                res = model2.predict(np.expand_dims(all_keypoints, axis=0))[0]
                # blokada po detekcji :)
                if actions2[np.argmax(res)] == sentence[-1] and stop_still_max:
                    if stop < 16:
                        stop = 16
                else:
                    stop_still_max = False

                # collecting 5 last words for top bar
                if res[np.argmax(res)] > threshold:

                    # last 5 frames must predict the same sign
                    if (last_one == actions2[np.argmax(res)]):
                        last_one_counter += 1
                    else:
                        last_one = actions2[np.argmax(res)]
                        last_one_counter = 1
                        stop_still_max = False

                    # blokada po detekcji :)
                    if actions2[np.argmax(res)] == sentence[-1] and stop_still_max:
                        stop = 20
                    else:
                        stop_still_max = False

                    if ((last_one_counter > 4 and counter_2 > 15) or (clean and last_one_counter > 1)):
                        # no_landmarks deleting
                        if (clean):
                            sentence = []
                            clean = False

                        if len(sentence) > 0:
                            if actions2[np.argmax(res)] != sentence[-1] and stop == 0:
                                print("Dodano znak, counter:", counter_2)
                                sentence.append(actions2[np.argmax(res)])
                                stop = 20
                                stop_still_max = True
                        else:
                            sentence.append(actions2[np.argmax(res)])
                            stop = 20
                            stop_still_max = True

                        last_nentwork = 2

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # showing probabilities for every word
                for num, prob in enumerate(res):
                    cv2.rectangle(image, (0, 60 + num * 30), (int(prob * 100), 90 + num * 30), colors2[num], -1)
                    cv2.putText(image, actions2[num], (0, 85 + num * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)




            # needed because of np concatenation
            all_keypoints = []

            # showing top bar
            cv2.rectangle(image, (0, 0), (640, 40), top_bar_color, -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255) , 2,
                        cv2.LINE_AA)

            # show to screen
            cv2.imshow('OpenCV ladnmark_test', image)

            # break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            print("---")
            print("")


    cap.release()
    cv2.destroyAllWindows()

    return {"message": " ".join(sentence)}

# --------------------------------------------------------------------------------------------------------









### API PART ###
@app.get("/")
async def root():
    return {"message": "Online!"}

@app.post("/video/test-asl")
def testASL(file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)
    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        res = processVideo(temp.name)
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)

    return res


@app.get("/{full_path:path}")
def createItem(full_path: str):
    # result = {"link": link+"dupa"}
    res = processVideo(full_path)

    return res

if __name__ == "__main__":
    init_global_variables_on_zero()
    port = int(os.environ.get("PORT", 5050))
    run(app, host="0.0.0.0", port=port)


# 2 5 przed ostatni

# INSTRUKCJA URUCHOMIENIA API
#
# W PRZEGLĄDARCE:
# 0.0.0.0:5050/ - STATUS CHECK
# 0.0.0.0:5050/{LINK} - PREDYKCJA
# 0.0.0.0:5050/docs - INTERAKTYWNA INSTRUKCJA API
#
# NA TELEFONIE:
# SPRAWDZIĆ ADRES IP SIECI
# W PRZEGLĄDARCE NA TELEFONIE ODPALIĆ {ADRES IP SIECI}:5050/
#
# PRZYKŁADOWY LINK:
# http://0.0.0.0:5050/https%3A%2F%2Ffirebasestorage.googleapis.com%2Fv0%2Fb%2Fasl-recognition-d264d.appspot.com%2Fo%2Fasl-test2.mp4%3Falt%3Dmedia%26token%3D0e930fb2-1bc6-4575-9dcf-5152132eb4f4
# W LINKU DO WIDEO NALEŻY ZAMIENIĆ ZNAKI SPECJALNE (NP / = ?) WEDŁUG SCHEMATU ZE STRONY:
# https://www.w3schools.com/tags/ref_urlencode.ASP
#
# HAVE FUN