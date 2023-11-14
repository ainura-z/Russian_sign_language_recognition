import matplotlib.pyplot as plt
import mediapipe as mp

def draw_landm_points(frame, points):
    """
    Draw landmarks(from annotation) on image

    Inputs:
        frame - frame of video
        points - dictionary of type filename:landmarks
    """
    height = frame.shape[0]
    width = frame.shape[1]
    
    x_coord = []
    y_coord = []
    for key in ['hand 1', 'hand 2']:
        if key in points.keys():
            coordx = [int(item['x']*width) for item in points[key]]
            coordy = [int(item['y']*height) for item in points[key]]
            x_coord.extend(coordx)
            y_coord.extend(coordy)
    
    plt.imshow(frame)
    plt.scatter(x_coord,y_coord,c='r',s=15)

def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
    """
    Draw landmarks(from mediapipe) on image

    Inputs:
        image - frame
        results - output from mediapipe detection
        mp_drawing - drawing tool
        mp_holistic - mediapipe tool
    """
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 