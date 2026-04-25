from src.utils.landmark_utils import detect_landmarks, hand_landmarker
from src.utils.keypoint_utils import extract_keypoints
import numpy as np


def predict_motion_pose_tf(sequence, model, actions):


    if len(sequence) == 30:
        model_input = np.expand_dims(np.array(sequence), axis=0)

        action_result = model.predict(model_input, verbose = 0)

        best_index = int(np.argmax(action_result))
        best_label = actions[best_index]
        confidence = np.max(action_result)
        
        return best_label, confidence 
    
    return None, 0.0

import onnxruntime as rt
import numpy as np
import sys, os

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

action_session = rt.InferenceSession(os.path.join(BASE_DIR, 'models', 'action_model.onnx'))
def predict_motion_pose(sequence, actions, threshold=0.8):
    if len(sequence) < 30:
        return None, 0.0
    
    model_input = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
    
    input_name = action_session.get_inputs()[0].name
    model_result = action_session.run(None, {input_name: model_input})[0]
    
    confidence = np.max(model_result)# type: ignore
    if confidence < threshold:
        return None, confidence
    
    best_index = np.argmax(model_result)# type: ignore
    best_label = actions[best_index]
    return best_label, confidence