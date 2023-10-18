import cv2

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