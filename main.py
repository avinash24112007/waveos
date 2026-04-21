import cv2


def capture_web_frames():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()

        cv2.imshow("Rewult", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


capture_web_frames()
