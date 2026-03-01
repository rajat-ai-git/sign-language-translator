import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
img = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
while True:
    isTrue, frame = img.read()
    

    if not isTrue:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)

    cv2.imshow('video', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
img.release()
cv2.destroyAllWindows()