import cv2, time, os
import mediapipe as mp
import numpy as np

from landmark_utils import detect_landmarks, draw_face_landmark_on_frame, draw_hand_landmark_on_frame, draw_pose_landmark_on_frame
from landmark_utils import face_landmarker, pose_landmarker, hand_landmarker

def extract_keypoints(hand_result):
    left_hand =np.zeros(63)
    right_hand = np.zeros(63)


    for i, hand_landmarn in enumerate(hand_result.hand_landmarks):

        label = hand_result.handedness[i][0].category_name
        cords = np.array([[res.x, res.y, res.z] for res in hand_result.hand_landmarks[i]]).flatten()

        if label == "Left":
            left_hand = cords
        else:
            right_hand = cords
    
    return np.concatenate([left_hand, right_hand])










def capture_keypoints_from_frames(DATA_PATH, actions, n_sequences, sequence_length, time_sep_seq,extract_kp: bool):
    cap = cv2.VideoCapture(0)

    for action in actions:
        for vid_seq_no in range(n_sequences):
            for frame_no in range(sequence_length):

                ret, frame = cap.read()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_frame = mp.Image(mp.ImageFormat.SRGB, data=rgb_frame)

                ts = int(time.time() * 1000)

                hand_result = detect_landmarks(mp_frame, hand_landmarker, ts)
                draw_hand_landmark_on_frame(frame, hand_result)

                # face_result = detect_landmarks(mp_frame, face_landmarker, ts)
                # draw_face_landmark_on_frame(frame, face_result)
                # pose_result = detect_landmarks(mp_frame, pose_landmarker, ts)
                # draw_pose_landmark_on_frame(frame, pose_result)
                
                if frame_no == 0:
                    cv2.putText(frame, "Starting Collection", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),4,  cv2.LINE_AA)
                    cv2.putText(frame, f"Collectiong frames for {action} Frame NO: {frame_no}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),1,  cv2.LINE_AA)
                    cv2.imshow("OpenCV feed", frame)    

                    cv2.waitKey(time_sep_seq)

                else:
                    cv2.putText(frame, f"Collectiong frames for {action} SeqNo {vid_seq_no} Frame NO: {frame_no}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1,  cv2.LINE_AA)
                    cv2.imshow("OpenCV feed", frame)

                if extract_kp:
                    keypoints = extract_keypoints(hand_result)
                    kp_path = os.path.join(DATA_PATH, action, str(vid_seq_no), str(frame_no))

                    np.save(kp_path, keypoints)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()