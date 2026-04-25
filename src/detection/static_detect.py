from src.utils.landmark_utils import detect_landmarks, hand_landmarker
from src.utils.keypoint_utils import extract_keypoints
import numpy as np


def predict_static_pose_tf(keypoints, model, actions):

    model_input = np.expand_dims(np.array(keypoints), axis=0)

    model_result = model.predict(model_input, verbose = 0)
    best_index = int(np.argmax(model_result))
    best_label = actions[best_index]
    confidence = np.max(model_result)
    
    return best_label, confidence 

import onnxruntime as rt
import numpy as np
import sys, os

BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

static_session = rt.InferenceSession(os.path.join(BASE_DIR, 'models', 'static_model.onnx'))
def predict_static_pose(keypoints, actions, threshold=0.9):
    model_input = np.expand_dims(np.array(keypoints), axis=0).astype(np.float32)
    
    input_name = static_session.get_inputs()[0].name
    model_result = static_session.run(None, {input_name: model_input})[0]
    
    confidence = np.max(model_result)# type: ignore
    if confidence < threshold:
        return None, confidence
    
    best_index = np.argmax(model_result)# type: ignore
    best_label = actions[best_index]
    return best_label, confidence
