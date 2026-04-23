from src.utils.landmark_utils import detect_landmarks, hand_landmarker
from src.utils.keypoint_utils import extract_keypoints
import numpy as np


def predict_motion_pose(sequence, model, actions):


    if len(sequence) == 30:
        model_input = np.expand_dims(np.array(sequence), axis=0)

        action_result = model.predict(model_input)

        best_index = int(np.argmax(action_result))
        best_label = actions[best_index]
        confidence = np.max(action_result)
        
        return best_label, confidence 
    
    return None, 0.0