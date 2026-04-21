
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

face_task_path = 'C:/Projects/Copmuter control by Human Gesture/assets/task_files/face_landmarker.task'
pose_task_path = 'C:/Projects/Copmuter control by Human Gesture/assets/task_files/pose_landmarker_lite.task'
hand_task_path = 'C:/Projects/Copmuter control by Human Gesture/assets/task_files/hand_landmarker.task'

face_landmarker_options = FaceLandmarkerOptions(BaseOptions(model_asset_path=face_task_path), running_mode = RunningMode.VIDEO)
hand_landmarker_options = HandLandmarkerOptions(BaseOptions(model_asset_path=hand_task_path), running_mode = RunningMode.VIDEO, num_hands=2)
pose_landmarker_options = PoseLandmarkerOptions(BaseOptions(model_asset_path=pose_task_path), running_mode = RunningMode.VIDEO)

face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
hand_landmarker = HandLandmarker.create_from_options(hand_landmarker_options)
pose_landmarker = PoseLandmarker.create_from_options(pose_landmarker_options)


def detect_landmarks(mp_frame,
                    landmarker, 
                    timestamp
                    ):
    return landmarker.detect_for_video(mp_frame, timestamp)

def draw_landmarks( frame,
                    points, 
                    connection_type
                   ):
    custom_point_spec = DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    custom_line_spec = DrawingSpec(thickness=1)

    mp_drawing.draw_landmarks(
        frame,
        points,
        connection_type,
        custom_point_spec,
        custom_line_spec
    )


def draw_hand_landmark_on_frame(frame, hand_result):

    if hand_result.hand_landmarks:
        for hand_landmark in hand_result.hand_landmarks:
            draw_landmarks(
                frame,
                hand_landmark,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            )


def draw_face_landmark_on_frame(frame, face_result):

    if face_result.face_landmarks:
        for face_landmark in face_result.face_landmarks:
            draw_landmarks(
                frame,
                face_landmark,
                mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
            )


def draw_pose_landmark_on_frame(frame, pose_result):

    if pose_result.pose_landmarks:
        for pose_landmark in pose_result.pose_landmarks:
            draw_landmarks(
                frame,
                pose_landmark,
                mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
            )