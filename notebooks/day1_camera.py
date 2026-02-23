import cv2
#import numpy as np
img = cv2.VideoCapture(0)
while True:
    isTrue, frame = img.read()

    if not isTrue:
        break

    #print(frame.shape)
    #print(grey.shape)

    cv2.rectangle(frame, (100, 100), (400,400), (0, 255, 0), thickness = 2)

    cv2.putText(frame, "Hello", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness = 1)
    cv2.imshow('Text', frame)




    if cv2.waitKey(1)&0xFF==ord('q'):
        break
img.release()
cv2.destroyAllWindows()
