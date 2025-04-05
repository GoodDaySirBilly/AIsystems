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

    cv2.circle(img, (23, 180), 20, GREEN, -1)
    cv2.circle(img, (400, 120), 20, YELLOW, -1)
    cv2.rectangle(img, (400, 300), (300, 200), RED, -1)
    cv2.rectangle(img, (50, 400), (100, 300), BLUE, -1)
    points = np.array([(100, 50), (50, 100), (150, 100)])
    cv2.drawContours(img, [points], 0, YELLOW, -1)

    rhombus_points = np.array([[260, 60], [360, 110], [260, 160], [160, 110]])
    cv2.drawContours(img, [rhombus_points], 0, GREEN, -1)

    trap = np.array([[300, 350], [350, 350], [370, 400], [280, 400]])
    cv2.drawContours(img, [trap], 0, RED, -1)

    tr_points = np.array([[300, 500], [400, 500], [350, 586]])
    cv2.drawContours(img, [tr_points], 0, GREEN, -1)

    pent_points = np.array([[500, 550], [570, 650], [650, 600], [650, 500], [550, 450]])
    cv2.drawContours(img, [pent_points], 0, RED, -1)

    trpr_points = np.array([[100, 600], [150, 650], [120, 700]])
    cv2.drawContours(img, [trpr_points], 0, BLUE, -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # alghorithm of decoloring
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # finding counters
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    colorInf = {}

    masks = { # dictionary where key = color, value = borders
        RED:    ((0  , 10 , 50 ), (10 , 255, 255)), 
        YELLOW: ((25 , 200, 200), (30 , 255, 255)),
        GREEN:  ((34 , 0  , 50 ), (75 , 255, 255)),
        BLUE:   ((101, 0  , 50 ), (128, 255, 255)),
    }

    colors = {
        RED    : "RED",
        YELLOW : "YELLOW",
        GREEN  : "GREEN",
        BLUE   : "BLUE"
    }

    for color in colors.keys():

        mask = cv2.inRange(hsv, masks[color][0], masks[color][1])

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        
        for cnt in contours:

            figureMoments = cv2.moments(cnt)

            centerX = int(figureMoments["m10"] / figureMoments["m00"])
            centerY = int(figureMoments["m01"] / figureMoments["m00"])

            colorInf[(centerX, centerY)] = colors[color]

    shapes = {}

    print("FIGURE TYPE      COLOR    SQUARE")
    print("----------------------------------")
    for c in cnts:

        figureMoments = cv2.moments(c)

        centerX = int((figureMoments["m10"] / figureMoments["m00"]))
        centerY = int((figureMoments["m01"] / figureMoments["m00"]))

        shape = detect(c, False)

        area = cv2.contourArea(c)

        cv2.drawContours(img, [c], -1, BLACK, 1) # drawing counters
        
        cv2.circle(img, (centerX, centerY), 5, BLACK, -1) # marking center

        detectedColor = colorInf.get(
            (centerX, centerY),
            "Undefined color"
        )  
        
        cv2.putText(img, shape + " " + detectedColor + " " + str(cv2.contourArea(c)),
                     (centerX, centerY), cv2.FONT_HERSHEY_COMPLEX, 0.5, BLACK, 2)
        formatArgs = {
            'Figure' : shape,
            'Color'  : detectedColor,
            'Square' : area
        }
        
        print("{Figure:<15}  {Color:<6}  {Square:<.2f} "
              .format(**formatArgs))

        if shape in shapes:
            shapes[shape] += 1
        else:
            shapes[shape] = 1


    text1 = f"I found {len(cnts)} objects!"
    text2 = "Sayfov R. K. 1242"


    cv2.putText(img, text1, (650, 900), cv2.FONT_ITALIC, 0.7, BLACK,2)
    cv2.putText(img, text2, (20, 45), cv2.FONT_ITALIC, 0.7, BLACK,2)
    
    cv2.imshow("Contours", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()