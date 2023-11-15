import cv2
import numpy as np
import mediapipe as mp
import torch
from model.landmarks import mediapipe_detection, extract_keypoints
from utils.visualization import draw_styled_landmarks, prob_viz


def main():
    # keypoint using MP Holistic
    mp_holistic = mp.solutions.holistic
    # drawing utilities
    mp_drawing = mp.solutions.drawing_utils

    # labels for detecting
    # labels = ['Привет!', 'Пока', 'Я', 'тебя', 'любить']
    labels = ["hi!", "bye", "I", "you", "love"]
    label_encoding = {label: i for i, label in enumerate(labels)}
    inv_lbl_encoding = {value: key for key, value in label_encoding.items()}
    # colors for drawing
    colors = [
        (245, 117, 16),
        (117, 245, 16),
        (16, 117, 245),
        (16, 117, 245),
        (16, 117, 245),
    ]

    # reading model
    model = torch.load("model.pt", map_location=torch.device("cpu"))

    # sequence of keypoints
    sequence = torch.tensor([[0] * 63])
    sentence = []
    # threshold for classification
    threshold = 0.7
    i = 0

    cap = cv2.VideoCapture(0)
    # set mediapipe model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            i += 1
            # frame = cv2.flip(frame, 1)
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results, mp_drawing, mp_holistic)

            # 2. Prediction logic
            keypoints = extract_keypoints(results).unsqueeze(0).float()
            #         sequence.insert(0,keypoints)
            #         sequence = sequence[:30]
            sequence = torch.cat((sequence, keypoints), dim=0)
            sequence = sequence[-32:]

            if len(sequence) == 32 and i % 32 == 0:
                res = torch.nn.functional.softmax(model(sequence.unsqueeze(0)))
                print(res)
                label = torch.max(res, 1)[1].item()
                print("Label:", inv_lbl_encoding[label])

                # visualization
                if abs(res[0][label].item()) > threshold:
                    if len(sentence) > 0:
                        if inv_lbl_encoding[label] != sentence[-1]:
                            sentence.append(inv_lbl_encoding[label])
                    else:
                        sentence.append(inv_lbl_encoding[label])
                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                    # cv2.putText(image, ' '.join(sentence), (3,30),
                    #        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if abs(res[0][label].item()) < 0.5:
                    sentence = [""]
                # visialization probabilities
                # image = prob_viz(res, inv_lbl_encoding, image, colors)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                " ".join(sentence),
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # show to screen
            cv2.imshow("Russian Sign Language Recognition", image)

            # break
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
