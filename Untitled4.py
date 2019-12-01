#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model 

import numpy as np 

import cv2 

model =load_model('mymodel.h5') 

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 

 

from skimage.transform import resize 

def detect1(frame): 

    try: 

        img= resize(frame,(64,64)) 

        img = np.expand_dims(img,axis=0) 

        if(np.max(img)>1): 

            img =img/255.0 

        prediction =model.predict(img) 

        prediction_class = model.predict_classes(img) 

        if (prediction_class[0] == 0): 

            data =0 

        if (prediction_class[0] == 1): 

            data =1 

        if (prediction_class[0] == 2): 

              data =2                                          

    except AttributeError: 

        print("shape not found") 

    return data 


# In[2]:


import time 

import sys 

import ibmiotf.application 

import ibmiotf.device 

import random 

import cv2 

data1 = "" 

#organization = "ei8unc" 

#deviceType = "varun" 

#deviceId = "24067515" 

#authMethod = "token" 

#authToken = "uRLF7x-1HVoPUTgnlS" 

#try: 

 #   deviceO
#ptions = {"org": organization, "type": deviceType, "id": deviceId, "auth-method": authMethod, "auth-token": authToken} 

  #  deviceCli = ibmiotf.device.Client(deviceOptions) 

#except Exception as e: 

 #   print("Caught exception connecting device: %s" % str(e)) 

  #  sys.exit() 

        # Connect and send a datapoint "hello" with value "world" into the cloud as an event of type "greeting" 10 times 

#deviceCli.connect() 

  

# Loading the cascades 

face_cascade = cv2.CascadeClassifier('C:/Users/akura/haarcascade_frontalface_default.xml') 

# Defining a function that will do the detections 

def detect(gray, frame): 

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    for (x, y, w, h) in faces: 

        print (faces.shape) 

        print ("Number of faces detected: " + str(faces.shape[0])) 

        print("Data Found") 

        cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 

        roi_color = frame[y:y+h, x:x+w] 

        FaceFileName = "face_" + str(y) + ".jpg" 

        cv2.imwrite(FaceFileName, roi_color) 

        img=cv2.imread(FaceFileName) 

        data=detect1(img) 

        if (data== 0): 

            cv2.putText(frame,"barak", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

            data1 = { 'person' : "barak"} 

            #success = deviceCli.publishEvent("DHT11", "json", data1, qos=0, on_publish=myOnPublishCallback) 

            time.sleep(0.5) 

            #deviceCli.disconnect() 

             

        if (data== 1): 

            cv2.putText(frame,"modi", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

            data1 = { 'person' : "modi"} 

            #success = deviceCli.publishEvent("DHT11", "json", data1) 

            time.sleep(0.5) 

            #deviceCli.disconnect() 

        if (data== 2): 

            cv2.putText(frame,"vladimier", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

            data1 = { 'person' : "vladimier"} 

            #success = deviceCli.publishEvent("DHT11", "json", data1, qos=0) 

            time.sleep(0.5) 

            #deviceCli.disconnect() 

  

    return frame 

  

# Doing some Face Recognition with the webcam 

video_capture = cv2.VideoCapture(0) 

while True: 

    _,frame = video_capture.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    canvas = detect(gray, frame) 

    cv2.imshow('Video', canvas) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 

        break 

video_capture.release() 

cv2.destroyAllWindows() 


# In[ ]:





# In[ ]:




