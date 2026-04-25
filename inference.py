# src/inference.py — only what's needed for prediction
from src.utils.landmark_utils import hand_landmarker, detect_landmarks, draw_hand_landmark_on_frame
from src.utils.keypoint_utils import extract_keypoints
from src.detection.action_detect import predict_motion_pose
from src.detection.static_detect import predict_static_pose
from src.commands.executor import execute_command, detect_change_gui
from src.commands.profiles import CHROME_PROFILE, SPOTIFY_PROFILE, DEFAULT_PROFILE

MOTION_ACTIONS = ['SWIPE_LEFT', 'SWIPE_RIGHT', 'CIRCLE', 'NEUTRAL']
STATIC_ACTIONS = ['FIST', 'OPEN_PALM', 'PEACE', 'NEUTRAL']