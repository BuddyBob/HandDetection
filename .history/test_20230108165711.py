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
        processed = hands.process(file)
        print('Hand Type: ', processed.multi_handedness)

