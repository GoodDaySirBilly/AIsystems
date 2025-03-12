def main():

    import cv2
    import numpy as np
    import imutils

    from tools import detect

    WHITE  = (255, 255, 255)
    RED    = (0, 0 ,255)
    GREEN  = (0, 255, 0)
    BLUE   = (255, 0, 0)
    YELLOW = (0, 255, 255)
    BROWN  = (101, 67, 33) # it must be brown
    VIOLET = (255, 0, 255)
    BLACK  = (0, 0, 0)

    img = np.full((1000, 1000, 3), WHITE, np.uint8) # creating white sc

    # creating figures 
    cv2.circle(img,(300,400), 40, RED, -1)

    cv2.rectangle(img, (800, 600), (600, 400), YELLOW, -1) # sqare

    cv2.rectangle(img, (100, 800), (200, 600), BLUE, -1) # rectangle

    triangle_points = np.array([(200, 100), (100, 200), (300, 200)])
    cv2.drawContours(img, [triangle_points], 0, BROWN, -1)

    rhombus_points = np.array([[520, 120], [720, 220], [520, 320], [320, 220]])
    cv2.drawContours(img, [rhombus_points],0, VIOLET, -1)

    trapezund_points = np.array([[600, 700], [700, 700],  [740, 800], [560, 800]])
    cv2.drawContours(img, [trapezund_points],0, GREEN, -1)
    # creating figures

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # alghorithm of decoloring
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # finding counters

    for c in cnts: # for every contour in contours

        figureMoments = cv2.moments(c) # taking moments(statistic of every spot)
        
        centerX = int((figureMoments["m10"] / figureMoments["m00"])) # medium arhithmetic vaue of X
        centerY = int((figureMoments["m01"] / figureMoments["m00"])) # _ of Y

        shape = detect(c) # using object of class ShapeDetector to detect

        cv2.drawContours(img, [c], -1, BLACK, 1) # drawing counters 
        cv2.circle(img, (centerX, centerY), 5, BLACK, -1) # marking center
        cv2.putText(img, shape, (centerX+20, centerY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1)
        
        
    text1 = f"I found {len(cnts)} objects!"
    text2 = "Sayfov R. K. 1242"


    cv2.putText(img, text1, (650, 900), cv2.FONT_ITALIC, 0.7, BLACK,2)
    cv2.putText(img, text2, (20, 45), cv2.FONT_ITALIC, 0.7, BLACK,2)
    
    cv2.imshow("Contours", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
