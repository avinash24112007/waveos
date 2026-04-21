import os

from src.utils.directory_creator import create_folders
from src.utils.keypoint_utils import capture_keypoints_from_frames


DATA_PATH = os.path.join("DATA_test", "STATIC_DATA")
actions = ["swipe_left", "swipe_right", "circle"]
n_seq = 10
print("Creating folders at:", DATA_PATH)

create_folders(
    DATA_PATH=DATA_PATH,
    actions=actions,
    n_sequences=10
)

capture_keypoints_from_frames(
    DATA_PATH=DATA_PATH,
    actions=actions,
    n_sequences=n_seq,
    sequence_length=30,
    time_sep_seq=2,
    extract_kp=False
)