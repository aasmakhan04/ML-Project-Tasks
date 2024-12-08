#Face Detection using Haar Cascades
import cv2
harcascade= "model/haarcascade_frontalface_default.xml"

#Accessing your camera
cap= cv2.VideoCapture(0)
cap.set(3, 500) #width
cap.set(3, 420) #height

while True:
    success, img = cap.read()
    facecascade= cv2.CascadeClassifier(harcascade)
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #setting the image color as gray
     
    face= facecascade.detectMultiScale(img_gray, 1.1,4)
    for (x,y,w,h) in face:   #creating a rectangle around your face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
     
    cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #closing the camera
      break
  
  
    
