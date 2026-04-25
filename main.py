import sys
import os

# Get the directory where the exe/script is located
if getattr(sys, 'frozen', False):
    # Running as PyInstaller exe
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as normal Python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


from src.collections.action_collect import action_collection_pipeline
from src.collections.static_collect import static_collection_pipeline
from sklearn.model_selection import train_test_split

from src.training.action_training import train_action_NN
from src.training.static_training import train_static_Conv1D_NN, train_static_Dense_NN, train_static_Conv1D_NN
from src.training.callbacks import early_stop, save_best_model

from src.utils.landmark_utils import hand_landmarker, detect_landmarks, draw_hand_landmark_on_frame
from src.utils.keypoint_utils import extract_keypoints
from src.utils.gui_utils import get_current_app

from src.detection.action_detect import predict_motion_pose
from src.detection.static_detect import predict_static_pose

from src.commands.executor import execute_command, detect_change_gui


from src.commands.profiles import CHROME_PROFILE, SPOTIFY_PROFILE, DEFAULT_PROFILE

from collections import deque

from src.utils.app_state import AppState, state

import cv2, time
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time

DATA_PATH_STATIC = "DATA/static"
DATA_PATH_ACTION = "DATA/action"
STATIC_ACTIONS = ['FIST', 'OPEN_PALM', 'PEACE', 'NEUTRAL']
MOTION_ACTIONS = ['SWIPE_LEFT', 'SWIPE_RIGHT', 'CIRCLE', 'NEUTRAL']


def pipeline():
    

    STATIC_X, STATIC_Y = static_collection_pipeline(DATA_PATH_STATIC, STATIC_ACTIONS, 100, 1, 500)
    assert STATIC_X is not None
    STATIC_X = STATIC_X.reshape(-1, 126)  
    ACTION_X, ACTION_Y = action_collection_pipeline(DATA_PATH_ACTION, MOTION_ACTIONS, 60, 30, 2000)

    STATIC_X_TRAIN, STATIC_X_TEST, STATIC_Y_TRAIN, STATIC_Y_TEST = train_test_split(STATIC_X, STATIC_Y)
    ACTION_X_TRAIN, ACTION_X_TEST, ACTION_Y_TRAIN, ACTION_Y_TEST = train_test_split(ACTION_X, ACTION_Y)


    ACTION_SAVE = save_best_model('models/action_model.keras')
    ACTION_CALLBACKS = [early_stop(), ACTION_SAVE]
    STATIC_SAVE1 = save_best_model('models/STATIC_model_t1.keras')
    STATIC_CALLBACKS1 = [early_stop(), STATIC_SAVE1]
    STATIC_SAVE2 = save_best_model('models/STATIC_model_t2.keras')
    STATIC_CALLBACKS2 = [early_stop(), STATIC_SAVE2]



    ACTION_MODEL = train_action_NN(X_Train=ACTION_X_TRAIN, Y_Train=ACTION_Y_TRAIN, epochs=1000, callback=ACTION_CALLBACKS, actions=MOTION_ACTIONS)
    STATIC_MODEL_1 = train_static_Dense_NN(X_TRAIN=STATIC_X_TRAIN, Y_TRAIN=STATIC_Y_TRAIN, epochs=1000, callback=STATIC_CALLBACKS1, actions=STATIC_ACTIONS)
    STATIC_MODEL_2 = train_static_Conv1D_NN(X_TRAIN=STATIC_X_TRAIN, Y_TRAIN=STATIC_Y_TRAIN, epochs=1000, callback=STATIC_CALLBACKS2, actions=STATIC_ACTIONS)

    return ACTION_MODEL, STATIC_MODEL_1, STATIC_MODEL_2




def predict(action_model, static_model, MOTION_ACTIONS, STATIC_ACTIONS):

    cap = cv2.VideoCapture(0)

    sequence = deque(maxlen=30)

    prev_kp = np.zeros(126)
    

    while cap.isOpened():
        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(mp.ImageFormat.SRGB, data=rgb_frame)

        ts = int(time.time() * 1000)
        hand_result = detect_landmarks(mp_frame, hand_landmarker, ts)

        prediction_kp = extract_keypoints(hand_result)

        movement = np.linalg.norm(prediction_kp - prev_kp)

        static_label, static_conf = None, 0.0
        motion_label, motion_conf = None, 0.0


        if movement > 0.1:
            sequence.append(prediction_kp)
            motion_label, motion_conf = predict_motion_pose(sequence, action_model, MOTION_ACTIONS)
        else:
            sequence.append(prediction_kp)
            static_label, static_conf = predict_static_pose(prediction_kp, static_model, STATIC_ACTIONS)

        prev_kp = prediction_kp

        draw_hand_landmark_on_frame(frame, hand_result)
        cv2.putText(frame, f"Static: {static_label} ({static_conf:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion: {motion_label} ({motion_conf:.2f})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Feed", frame)

        
        current_app = detect_change_gui() # Detect for every frame or have a cooldown
        match current_app:
            case 'edge' | 'chrome':
                execute_command(static_label, CHROME_PROFILE)
                execute_command(motion_label, CHROME_PROFILE)
            case 'spotify':
                execute_command(static_label, SPOTIFY_PROFILE)
                execute_command(motion_label, SPOTIFY_PROFILE)
            case _:
                execute_command(static_label, DEFAULT_PROFILE)
                execute_command(motion_label, DEFAULT_PROFILE)

        print(current_app)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# a, b, c = pipeline()


action_model = load_model('models/action_model.keras')
static_model = load_model('models/STATIC_model_t2.keras')



if __name__ == "__main__":
    predict(action_model, static_model, MOTION_ACTIONS, STATIC_ACTIONS)
