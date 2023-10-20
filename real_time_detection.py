import cv2
import numpy as np
import mediapipe as mp
import torch
from matplotlib import pyplot as plt
from data.data_utils import mediapipe_detection, extract_keypoints

# Keypoint using MP Holistic

# Holistic model
mp_holistic = mp.solutions.holistic 
# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# labels for detecting
# labels = ['Привет!', 'Пока', 'Я', 'тебя', 'любить']
labels = ['hi!', 'bye', 'me', 'you', 'love']
label_encoding = {label:i for i, label in enumerate(labels)}
inv_lbl_encoding = {value:key for key, value in label_encoding.items()}


# reading model
model = torch.load('model.pt', map_location=torch.device('cpu'))


# New detection variables
sequence = torch.tensor([[0]*126])
sentence = []
threshold = 0.2

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results).unsqueeze(0)
        sequence = torch.cat((sequence,keypoints), dim=0)
        sequence = sequence[-32:]
        
        if len(sequence) == 32:
            print(np.array(sequence).shape)
            res = model(sequence.unsqueeze(0))
            label = torch.max(res, 1)[1].item()
            print(inv_lbl_encoding[label])
            
            
        #3. Viz logic
            if abs(res[0][label].item()) > threshold: 
                if len(sentence) > 0: 
                    if inv_lbl_encoding[label] != sentence[-1]:
                        sentence.append(inv_lbl_encoding[label])
                else:
                    sentence.append(inv_lbl_encoding[label])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




