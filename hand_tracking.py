import cv2
import mediapipe as mp
import time

# Setting up the webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
# Create a "hands" object
# Set things to default values. This means static_mage_mode becomes false
# This means that we don't have to keep constantly detecting the hands, decreasing performance
# As long as the tracking confidence is high, no detection is needed
hands = mp_hands.Hands()

# Getting drawing utilities for easy hand drawing
mp_draw = mp.solutions.drawing_utils

prev_time = 0
current_time = 0

# Continually show webcam frames
while True:
    success, img = cap.read()
    # Convert img to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    # If several hands are detected, loop through them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing landmarks and connections on img
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculating the FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Drawing the FPS onto img
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.waitKey(1)
