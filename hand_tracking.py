import cv2
import mediapipe as mp
import time

# Setting up the webcam
cap = cv2.VideoCapture(0)

# Create a "hands" object
mpHands = mp.solutions.hands
# Set things to default values. This means static_mage_mode becomes false
# This means that we don't have to keep constantly detecting the hands, decreasing performance
# As long as the tracking confidence is high, no detection is needed
hands = mpHands.Hands()

# Continually show webcam frames
while True:
    success, img = cap.read()
    # Convert img to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
