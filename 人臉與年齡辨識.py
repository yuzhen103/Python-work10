#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/UESR/Desktop/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/UESR/Desktop/haarcascade_eye.xml')

img = cv2.imread('C:/Users/UESR/Desktop/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[7]:


import random
font = cv2.FONT_HERSHEY_SIMPLEX
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img, str(random.randrange(20, 30)), (x+(w//2)-18, y-10), font, 1, (14, 201, 255), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


import sys, cv2
imagePath = ('C:/Users/UESR/Desktop/23.jpg')
image = cv2.imread(imagePath)


# In[46]:


print(image.shape)


# In[47]:


plt.imshow(image)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# In[48]:


face_cascade = cv2.CascadeClassifier('C:/Users/UESR/Desktop/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE)

print(faces)


# In[49]:


import random
font = cv2.FONT_HERSHEY_SIMPLEX

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (14, 201, 255), 2)
    cv2.putText(image, str(random.randrange(30, 40)), (x+(w//2)-18, y-10), font, 1, (14, 201, 255), 2)
    cv2.rectangle(image, (image.shape[1]-140, image.shape[0]-20),
                 (image.shape[1], image.shape[0]), (0, 255, 255), -1)
    cv2.putText(image, "Finding  " + str(len(faces)) + " face  ", 
               (image.shape[1]-135, image.shape[0]-5),
                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0), 1) 

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# In[58]:


from PIL import Image
num = 1
for (x, y, w, h) in faces:
    filename = "face" + str(num) + ".jpg"
    image = Image.open("C:/Users/UESR/Desktop/23.jpg")
    imageCrop = image.crop((x, y, x+w, y+h))
    imageResize = imageCrop.resize((150, 150)) 
    imageResize.save(filename)
    num +=1


# In[59]:


face1 = cv2.imread('C:/Users/UESR/face1.jpg')
plt.imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))


# In[60]:


face2 = cv2.imread('C:/Users/UESR/face2.jpg')
plt.imshow(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))


# In[61]:


face3 = cv2.imread('C:/Users/UESR/face3.jpg')
plt.imshow(cv2.cvtColor(face3, cv2.COLOR_BGR2RGB))


# In[62]:


face4 = cv2.imread('C:/Users/UESR/face4.jpg')
plt.imshow(cv2.cvtColor(face4, cv2.COLOR_BGR2RGB))


# In[63]:


face5 = cv2.imread('C:/Users/UESR/face5.jpg')
plt.imshow(cv2.cvtColor(face5, cv2.COLOR_BGR2RGB))


# In[ ]:




