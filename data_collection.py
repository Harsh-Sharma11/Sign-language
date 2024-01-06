import os 
import cv2
import mediapipe as mp
import numpy as np
import time 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Space']
width, height = 640, 480
blank_image = cv2.imread("C:\\Users\\harsh\\Desktop\\5th sem\\project\\attempt 2\\white.jpg")




for k in labels:
     os.makedirs(k)
     frame_count = 0
     for m in range (10000):
        
        data_aux = []
        x_ = []
        y_ = []
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read a frame from the camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_with_landmarks = np.copy(blank_image) 

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_with_landmarks,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
        
        cv2.imshow('frame', frame)

        frame_copy = np.copy(frame_with_landmarks)
        frame_count += 1
        output_filename = os.path.join(k, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_filename, frame_copy)
        print("Saved frame:", frame_count, "for label:", k)
         
        if cv2.waitKey(1) & 0xFF == 27:
            break

    
    

cap.release()
cv2.destroyAllWindows()
