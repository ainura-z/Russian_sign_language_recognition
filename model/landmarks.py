import cv2
import torch


def mediapipe_detection(image, model):
    """
    Detection of landmarks with Mediapipe

    Inputs:
        image - frame of video
        model - mediapipe model

    Returns:
        image - frame
        results - mediapipe object of detected landmarks
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """
     Function for extracting keypoints from results

    Inputs:
        results - all detected keypoints after model

    Returns:
        concatenated vector of hsnd keypoints
    """
    # extracting left hand keypoints
    lh = (
        torch.tensor(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else torch.tensor([0] * 21 * 3)
    )
    # extracting right hand keypoints
    rh = (
        torch.tensor(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else torch.tensor([0] * 21 * 3)
    )

    return torch.concatenate([rh])
