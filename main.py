import cv2 as cv
import numpy as np
from keras.models import model_from_json


emotion_dict={0:"Angry",1:"Disgusting",2:"Fear",3:"Happy",4:"Neutral",5:"Sad",6:"Surprise"}

json_file=open("emotion_model.json","r")
loaded_model_json=json_file.read()
json_file.close()
emotion_model=model_from_json(loaded_model_json)

emotion_model.load_weights("emotion_model.h5")
print("Loaded model form disk")

cap=cv.VideoCapture("Demo_video.mp4") # put 0 for front camera

while True:
    res,frame=cap.read()
    frame=cv.flip(frame,1)
    frame=cv.resize(frame,(800,600))

    if not res:
        break

    face_detector=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)


    num_faces=face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors=5)

    for x,y,w,h in num_faces:
        cv.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame=gray_frame[y:y+h,x:x+w]
        cropped_image=np.expand_dims(np.expand_dims(cv.resize(roi_gray_frame,(48,48)),-1),0)

        # predict emotion
        emotion_prediction=emotion_model.predict(cropped_image)
        max_index=int(np.argmax(emotion_prediction))
        # print(max_index)

        cv.putText(frame,emotion_dict[max_index],(x+5,y-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)


    cv.imshow("Emotion Detection",frame)
    if cv.waitKey(1)==27 or cv.waitKey(1)==ord("q"):
        break

cap.release()
cv.destroyAllWindows()





