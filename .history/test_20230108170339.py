import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


IMAGE_FILES = []
img = cv2.imread('hand1.jpeg')
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
        print('Hand Landmarks: ', processed.multi_hand_landmarks)
        image_height, image_width, _ = file.shape
        annotated_image = file.copy()
        for hand_landmark in processed.multi_hand_landmarks:
            print(mp_hands.HAND_CONNECTIONS)
            

