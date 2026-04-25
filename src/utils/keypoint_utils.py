
# from keras.utils import to_categorical
import cv2, time, os
import mediapipe as mp
import numpy as np

from src.utils.landmark_utils import detect_landmarks, draw_face_landmark_on_frame, draw_hand_landmark_on_frame, draw_pose_landmark_on_frame
from src.utils.landmark_utils import hand_landmarker

def extract_keypoints(hand_result):
    left_hand =np.zeros(63)
    right_hand = np.zeros(63)


    for i, hand_landmark in enumerate(hand_result.hand_landmarks):

        label = hand_result.handedness[i][0].category_name
        cords = np.array([[res.x, res.y, res.z] for res in hand_result.hand_landmarks[i]]).flatten()

        if label == "Left":
            left_hand = cords
        else:
            right_hand = cords
    
    return np.concatenate([left_hand, right_hand])










def capture_keypoints_from_frames(DATA_PATH, actions, n_sequences, sequence_length, time_sep_seq,extract_kp: bool):
    cap = cv2.VideoCapture(0)

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    break_loop = False

    for action in actions:
        for vid_seq_no in range(n_sequences):
            window = []
            for frame_no in range(sequence_length):

                ret, frame = cap.read()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_frame = mp.Image(mp.ImageFormat.SRGB, data=rgb_frame)

                ts = int(time.time() * 1000)

                hand_result = detect_landmarks(mp_frame, hand_landmarker, ts)
                draw_hand_landmark_on_frame(frame, hand_result)

                if frame_no == 0:
                    cv2.putText(frame, "Starting Collection", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),4,  cv2.LINE_AA)
                    cv2.putText(frame, f"Collectiong frames for {action} SeqNo {vid_seq_no} Frame NO: {frame_no}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2,  cv2.LINE_AA)
                    cv2.imshow("OpenCV feed", frame)    

                    cv2.waitKey(time_sep_seq)

                else:
                    cv2.putText(frame, f"Collectiong frames for {action} SeqNo {vid_seq_no} Frame NO: {frame_no}", (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2,  cv2.LINE_AA)
                    cv2.imshow("OpenCV feed", frame)

                if extract_kp:
                    keypoints = extract_keypoints(hand_result)

                    kp_path = os.path.join(DATA_PATH, str(action), str(vid_seq_no), str(frame_no))

                    np.save(kp_path, keypoints)

                    window.append(keypoints)
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break_loop = True
                    break
            
            if break_loop:
                break

            if extract_kp:
                sequences.append(window)
                labels.append(label_map[action])

        if break_loop:
            break
    cap.release()
    cv2.destroyAllWindows()

    
    if extract_kp:
        X = np.array(sequences)
        Y = np.array(to_categorical(labels).astype(int)) # type: ignore
        print("Input Shape: ", X.shape)
        print("Output Shape: ", Y.shape)
        return X, Y
    else:
        return None, None

