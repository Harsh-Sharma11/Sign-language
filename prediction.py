# import os
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# cap = cv2.VideoCapture(0)
# model = keras.models.load_model("my_landmark_signlang_model.h5")

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'NEXT']
# width, height = 640, 480
# blank_image = np.zeros((height, width, 3), np.uint8)
# blank_image.fill(255)

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Could not read a frame from the camera.")
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_with_landmarks = np.copy(blank_image)

#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame_with_landmarks,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#     cv2.imshow('frame', frame)

#     frame_copy = np.copy(frame_with_landmarks)

#     # Preprocess the frame_copy image
#     frame_copy = cv2.resize(frame_copy, (480,640))  # Resize to match the model's input shape
#     frame_copy = frame_copy / 255.0  # Normalize the image (assuming the model expects values in [0, 1])

#     # Make a prediction
#     prediction = model.predict(np.expand_dims(frame_copy, axis=0))
#     print(prediction)
#     predicted_label_index = np.argmax(prediction)
#     print(predicted_label_index)
#     predicted_label = labels[predicted_label_index]
#     print(f"Predicted label: {predicted_label}")
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers


