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

# Time variables for calculating FPS
prev_time = 0
current_time = 0

# Continually show webcam frames
while True:
    success, img = cap.read()
    # Convert img to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    # If hands are detected, loop through them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Going through every landmark, with id starting from 0
            for id, landmark in enumerate(hand_landmarks.landmark):
                # print(id, landmark)

                # The coordinates of the landmarks will be returned as ratios of the image
                # This means we need to multiply values by the height and width of the image
                # This is so that we can get precise pixel locations
                height, width, channels = img.shape
                centre_x, centre_y = int(landmark.x * width), int(landmark.y * height)
                print("ID: " + str(id) + " X: " + str(centre_x) + " Y: " + str(centre_y))

                # Testing by drawing circles on the specified landmarks
                # 0 is the bottom of the hand, 4 is the tip of the thumb
                # if id == 0:
                #     cv2.circle(img, (centre_x, centre_y), 25, (255, 0, 255), cv2.FILLED)
                # elif id == 4:
                #     cv2.circle(img, (centre_x, centre_y), 25, (255, 0, 255), cv2.FILLED)

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
