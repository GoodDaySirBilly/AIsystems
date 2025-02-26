import cv2 as cv
import numpy as np
import imutils

white_list = np.zeros((1080,1920,3), dtype=np.uint8) + 0xff

#cv.rectangle(white_list, (70, 60), (250, 150), (0x09, 0xff, 0), -1)
cv.rectangle(white_list, (157, 289), (40, 560), (0, 0x14, 0), -1)
cv.circle(white_list, (700, 300), 70, (0xf0, 0xff, 0), -1)
cv.circle(white_list, (800, 500), 120, (0xBC, 0xA0, 0), -1)
#cv.rectangle(white_list, (65, 80), (90, 60), (0, 0, 0xFA), -1)
#cv.rectangle(white_list, (2, 2), (70, 90), (0xf0, 0x0f, 0), -1)


gray = cv.cvtColor(white_list, cv.COLOR_BGR2GRAY)

thresh = cv.threshold(gray, 0xff, 0xff, cv.THRESH_BINARY_INV)[1]

cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
output = white_list.copy()

for c in cnts:
    cv.drawContours(output, [c], -1, (240, 0, 159), 3)

cv.putText(
    output,
    f"Sayfov Rustam 1242 and I found {len(cnts)} objects",
    (80, 25),
    cv.FONT_HERSHEY_SIMPLEX,
    0.8, (0, 0, 0), 2
)


cv.imshow("Lab1AI", output)
cv.waitKey(0)
cv.destroyAllWindows()