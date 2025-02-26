import numpy as np
import cv2
import imutils

# Create a black image
img = np.zeros((1920,1080,3), np.uint8)

for i in range(1920):
    for j in range(1080):
        img[i][j] = (255,255,255)

# Draw a diagonal blue line with thickness of 5 px
#cv2.line(img,(0,0),(511,511),(255,0,0),5)

cv2.rectangle(img,(1920//2,1080//2),(510,128),(0,255,0),3)
cv2.circle(img,(1920//3,1080//3), 63, (0,0,255), -1)
cv2.ellipse(img,(256,256),(100,100),0,0,360,255,-1)

#pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
#pts = pts.reshape((-1,1,2))
#cv2.polylines(img,[pts],True,(0,255,255))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,f'1242 Sayfov Rustam Karomatulloevich found {len(cnts)}',(10,50), font, 1,(0,0,0),2,cv2.LINE_AA)

cv2.imshow("Image", img)
cv2.waitKey(1000000)