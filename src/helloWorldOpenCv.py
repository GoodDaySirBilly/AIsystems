import cv2
import numpy as np
import imutils

img = np.zeros((1080,1920,3), dtype=np.uint8) + 0xff

#cv2.rectangle(img, (70, 60), (250, 150), (0x09, 0xff, 0), -1)
cv2.rectangle(img, (157, 289), (40, 560), (0, 0x14, 0), -1)
cv2.circle(img, (700, 300), 70, (0xf0, 0xff, 0), -1)
cv2.circle(img, (800, 500), 120, (0xBC, 0xA0, 0), -1)
#cv2.rectangle(img, (65, 80), (90, 60), (0, 0, 0xFA), -1)
#cv2.rectangle(img, (2, 2), (70, 90), (0xf0, 0x0f, 0), -1)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

cv2.putText(
    output,
    f"Sayfov Rustam 1242 and I found {len(cnts)} objects",
    (80, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8, (0, 0, 0), 2
)


cv2.imshow("Lab1AI", output)
cv2.waitKey(0)
cv2.destroyAllWindows()