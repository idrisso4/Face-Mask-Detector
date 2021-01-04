import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model("face_mask.h5")
label = {0:"With Mask",1:"Without Mask"}
color_label = {0: (0,255,0),1 : (0,0,255)}
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frame = cv2.VideoCapture(0)

while(True):
	ret,img=frame.read();
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = cascade.detectMultiScale(gray, 1.1,4)
	for x,y,w,h in faces:
		face_image = img[y:y+h,x:x+w]
		resize_img  = cv2.resize(face_image,(150,150))
		normalized = resize_img/255.0
		reshape = np.reshape(normalized,(1,150,150,3))
		reshape = np.vstack([reshape])
		result = model.predict_classes(reshape)
		if result == 0:
			cv2.rectangle(img,(x,y),(x+w,y+h),color_label[0],3)
			cv2.rectangle(img,(x,y-50),(x+w,y),color_label[0],-1)
			cv2.putText(img,label[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
		elif result == 1:
			cv2.rectangle(img,(x,y),(x+w,y+h),color_label[1],3)
			cv2.rectangle(img,(x,y-50),(x+w,y),color_label[1],-1)
			cv2.putText(img,label[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
	cv2.imshow('img',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cam.release()
cv2.destroyAllWindows()
