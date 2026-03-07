import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import pickle

with open('model.pkl', 'rb')as f:
    model = pickle.load(f)


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
        row = []

        for hand in results.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                z = int(landmark.z * frame.shape[1])
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
                cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
               
        prediction = model.predict([row])
        cv2.putText(frame, str(prediction[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow('video', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
img.release()
cv2.destroyAllWindows()

