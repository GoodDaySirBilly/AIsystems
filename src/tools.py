import cv2
import numpy as np

def angle (a1: float, a2: float, a3: float):
    vec1 = a1 - a2
    vec2 = a3 - a2
    return np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))

def doubleEqual(a: float, b: float, delta = 1e-5):
    if abs(a-b) < delta:
        return True
    return False

def detect(c, printAngle: bool):
    shape = "Unidentified"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 3:

        vertices = approx.reshape(-1, 2)

        side1 = np.linalg.norm(vertices[0] - vertices[1])
        side2 = np.linalg.norm(vertices[1] - vertices[2])
        side3 = np.linalg.norm(vertices[2] - vertices[0])

        if (doubleEqual(side1,side2) and doubleEqual(side2,side3) and doubleEqual(side1,side3)):
            shape = "3 S Eq Triangle"
        elif (doubleEqual(side1,side2) or doubleEqual(side2,side3) or doubleEqual(side1,side3)):
            shape = "2 S Eq Triangle"
        else:
            shape = "Triangle"

    elif len(approx) == 4:

        (x, y, w, h) = cv2.boundingRect(approx)
        
        ar = w / float(h)

        vertices = approx.reshape(-1, 2)

        side1 = np.linalg.norm(vertices[0] - vertices[1])
        side2 = np.linalg.norm(vertices[1] - vertices[2])
        side3 = np.linalg.norm(vertices[2] - vertices[3])
        side4 = np.linalg.norm(vertices[3] - vertices[0])

        ar2 = float(side1)/float(side2)
        ar21 = float(side2)/float(side3)

        vertices = approx.reshape(-1, 2)

        angle1 = angle(vertices[3], vertices[0], vertices[1])
        angle2 = angle(vertices[0], vertices[1], vertices[2])
        angle3 = angle(vertices[1], vertices[2], vertices[3])
        angle4 = angle(vertices[2], vertices[3], vertices[0])

        ar3 = 90/angle1
        ar31 = 90/angle2
        ar32 = 90/angle3

        ar4 = float (angle1)/ float (angle2)
        ar41 = float (angle3)/ float (angle4)

        if (ar3 >= 0.95 and ar3 <= 1.05) and (ar31 >= 0.95 and ar31 <= 1.05) and (ar32 >= 0.95 and ar32 <= 1.05):
            if (ar >= 0.95 and ar <= 1.05):
                shape = "Square"
            else:
                shape = "Rectangle"
        elif (ar2 >= 0.95 and ar2 <= 1.05) and (ar21 >= 0.95 and ar21 <= 1.05):
            shape = "Romb"
        elif ((ar4 >= 0.95 and ar4 <= 1.05) and (ar41 >= 0.95 and ar41 <= 1.05)):
            shape = "Trapezund"
        
        if printAngle:
            print (f"{shape} angles:\n\
                    1: {str(angle1)[0:5]}\
                    2: {str(angle2)[0:5]}\
                    3: {str(angle3)[0:5]}\
                    4: {str(angle4)[0:5]}")
        
    elif len(approx) == 5:
        shape = "Pentagon"
    else:
        shape = "Circle"
        

    return shape
       
