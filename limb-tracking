import cv2
import mediapipe as mp

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils

# Open a video file for input
cap = cv2.VideoCapture('/Users/lauren/Downloads/Levis-Will-UKProDay.mp4')

# Get the video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

# Process each frame in the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If the frame is not valid, exit the loop
    if not ret:
        break
    
    # Detect the pose landmarks in the frame
    results = pose.process(frame)
    
    # Draw the pose landmarks and skeleton on the frame
    if results.pose_landmarks is not None:
        # Change the line thickness and color
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5))
    
    # Write the processed frame to the output video
    out.write(frame)
    
    # Display the frame with limb tracking
    cv2.imshow('Limb Tracking', frame)
    
    # Wait for a key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the input and output video objects and close the window
cap.release()
out.release()
cv2.destroyAllWindows()
