from src.utils.landmark_utils import detect_landmarks, hand_landmarker
from src.utils.keypoint_utils import extract_keypoints
import numpy as np


def predict_static_pose(keypoints, model, actions):

    model_input = np.expand_dims(np.array(keypoints), axis=0)

    model_result = model.predict(model_input)
    best_index = int(np.argmax(model_result))
    best_label = actions[best_index]
    confidence = np.max(model_result)
    
    return best_label, confidence 
