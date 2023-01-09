import cv2
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


IMAGE_FILES = []
for f in os.listdir('./HandInp'): 
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    IMAGE_FILES.append(img)

with mp_hands.Hands(static_image_mode=True, max_num_hands=1,min_detection_confidence=0.5) as hands:
    for i, file in enumerate(IMAGE_FILES):
        file = cv2.flip(file, 1)
        processed = hands.process(file)
        print('Hand Type: ', processed.multi_handedness)
        if not processed.multi_hand_landmarks:
            print("No hands detected")
            continue
        image_height, image_width, _ = file.shape
        annotated_image = file.copy()
        for hand_landmark in processed.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image,hand_landmark,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
            cv2.imwrite(f'./HandOut/annotated {i}', cv2.flip(annotated_image, 1))

