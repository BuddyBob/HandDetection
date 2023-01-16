import cv2
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



    
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2,min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No camera detected")
            continue
        
        image.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = True
        
        processed = hands.process(img)
        
        if not processed.multi_hand_landmarks:
            print("No hands detected")
            continue
        
        print('Hand Type: ', processed.multi_handedness)
        
        
        for hand_landmark in processed.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img,hand_landmark,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('MediaPipe Hands', cv2.flip(img, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


