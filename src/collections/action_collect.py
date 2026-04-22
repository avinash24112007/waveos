import os

from src.utils.directory_creator import create_folders
from src.utils.keypoint_utils import capture_keypoints_from_frames


DATA_PATH = os.path.join("DATA_test", "STATIC_DATA")
actions = ["swipe_left", "swipe_right", "circle"]
n_seq = 10
print("Creating folders at:", DATA_PATH)



def action_collection_pipelinr(DATA_PATH, actions, n_sequences, sequence_length, time_sep_seq):
    create_folders(
        DATA_PATH=DATA_PATH,
        actions=actions,
        n_sequences=n_sequences
    )

    X, Y = capture_keypoints_from_frames(
        DATA_PATH=DATA_PATH,
        actions=actions,
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        time_sep_seq=time_sep_seq,
        extract_kp=True
    )

    return X, Y