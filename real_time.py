#import openCv Library
import cv2

#get Video streaming from Camera
video_capture = cv2.VideoCapture(0)

#import Cascade Classifier for Frontal Face and Eye
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Loop for Video Streaming, Frame by Frame
while True:
	#Read Video Frame by Frame
	ret, frame = video_capture.read()
	#Resize Video by 50%
	small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
	#Prepare for Face Detection
	gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	#create Rectangle for each Face
	for (x,y,w,h) in faces:
		cv2.rectangle(small_frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = small_frame[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		#create Rectangle for each Eye
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	#show live Video Streaming
	cv2.imshow('Video', small_frame)
	#Press q for close
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv2.destroyAllWindows()
