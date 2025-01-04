import cv2
import mediapipe as mp
import math
import numpy as np

# Video Capture setup
vid = cv2.VideoCapture(0)
vid.set(3, 1280)

# MediaPipe Hands setup
mphands = mp.solutions.hands
Hands = mphands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mpdraw = mp.solutions.drawing_utils

# Initialize variables
drawing = False  # Track whether to draw
canvas = None  # Canvas to store drawings
prev_point = None

PINCH_TRESH = 75
DRAW_COLOR = (200, 200, 200)

# Pinch Detection and Drawing
def pinch(Tx, Ty, Px, Py):
    global drawing, prev_point
    T, P = [Tx, Ty], [Px, Py]
    dist = math.dist(T, P)
    if dist < PINCH_TRESH:
        drawing = True
        center_x = (Tx + Px) // 2
        center_y = (Ty + Py) // 2
        return center_x, center_y
    else:
        prev_point = None
        drawing = False
        return None

# Main loop
while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Mirror frame horizontally and convert to RGB
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = Hands.process(RGBframe)
    
    # Initialize the canvas on the first frame
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Detect hands and process landmarks
    if result.multi_hand_landmarks:
        for handLm in result.multi_hand_landmarks:
            for id, lm in enumerate(handLm.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Track thumb (id 4) and index finger (id 8)
                if id == 4:
                    Tx, Ty = cx, cy
                if id == 8:
                    center = pinch(Tx, Ty, cx, cy)
                    if center and drawing:
                        # Draw on the canvas
                        if prev_point:
                            cv2.line(canvas, prev_point, center, DRAW_COLOR, 5)
                        cv2.circle(canvas, center, 5, DRAW_COLOR, -1)
                        prev_point = center
    
    # Combine the canvas with the frame
    combined = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)
    
    # Display the combined frame
    cv2.imshow("Pinch Drawing", combined)
    
    # Break on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear the canvas when 'c' is pressed
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Release resources
vid.release()
cv2.destroyAllWindows()
