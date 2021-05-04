import pyautogui
from cv2 import cv2
import mediapipe as mp
# import time
# from  numpy a
import numpy as np


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
)


def findDistanceOfIndividualFingers(points, cors):
    first = cors[points[1]]
    sec = cors[0]
    distance = int(
        (((sec[1]-first[1])**2)+((sec[0]-first[0])**2))**(0.5))
    return distance


message = " Open Your index finger inside the Box "

mpDraw = mp.solutions.drawing_utils
fingertips = {"thumb": [2, 4], "index": [6, 8], "middle": [
    10, 11], "ring": [14, 16], "pinky": [18, 20], }
start_point = (50, 40)
pyautogui.FAILSAFE = False
end_point = (400, 280)
h1 = 395
w1 = 395
screen_width = pyautogui.size().width
screen_height = pyautogui.size().height
# vert_scale=
while True:
    _, img = cap.read()
    x, y = (pyautogui.position())
    cv2.putText(img, message, (20, 30),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)
    # cv2.rectangle(img, start_point, end_point, (255, 0, 0, 1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRgb)
    if results.multi_hand_landmarks:
        hands_list = []
        for handLndms in (results.multi_hand_landmarks):
            hand = {}
            for id, lm in enumerate(handLndms.landmark):
                h, w, c = img.shape
                hx, hy = int(lm.x*w), int(lm.y*h)
                hand[id] = (hx, hy)
            hands_list.append(hand)
            mpDraw.draw_landmarks(img, handLndms, mpHands.HAND_CONNECTIONS)
        filteredfingers = [(findDistanceOfIndividualFingers(fingertips[fingerkeys], hand))
                           < 150 for fingerkeys in fingertips.keys()]
        closedfingers = [
            filteredfinger for filteredfinger in filteredfingers if not filteredfinger]
        indexfinger = hand[8]
        cv2.circle(img, indexfinger, 5, (255, 0, 0), -1)
        wrist = hand[0]
        distanceboxSTART_with_index_finger = int(
            (((indexfinger[1]-start_point[1])**2)+((indexfinger[0]-start_point[0])**2))**(0.5))
        figer_inside_box = (not distanceboxSTART_with_index_finger > 400)
        indexfingerOpened = int(
            (((wrist[1]-indexfinger[1])**2)+((wrist[0]-indexfinger[0])**2))**(0.5)) > 200
        finger_inside_box = (
            indexfinger[0] > start_point[0] and indexfinger[0] < end_point[0])
        if(figer_inside_box):
            message = ""
            x_value = int(np.interp(indexfinger[0], [
                          start_point[0], end_point[0]], [0, screen_width]))
            y_value = int(np.interp(indexfinger[1], [
                          start_point[1], end_point[1]], [0, screen_height]))
            if len(filteredfingers) == 5:
                if(not (filteredfingers[1])):
                    pyautogui.moveTo(x_value, y_value)
                    if len(closedfingers)==2:
                        # print("2")
                        pyautogui.mouseDown(x_value, y_value)
                    else:
                        pyautogui.mouseUp(x_value, y_value)
                        
        else:
            message = " Open Your index finger inside the Box "

            cv2.putText(img, message, (20, 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)
    cv2.rectangle(img, start_point, end_point, (255, 255, 0), 2)
    cv2.imshow(" original ", img)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
