import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Getting the video
angle_min = []
angle_min_hip = []
cap = cv2.VideoCapture("squats.mp4")


# Curl counter variables
counter = 0
min_ang = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0
stage = None

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (640, 480)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Close the program when the video ends
        if not ret:
            break

        if frame is not None:
            frame_ = rescale_frame(frame, percent=75)

        # Recolor image to RGB
        image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]

            elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]
            knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]
            ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            ]

            # Get the landmarks for the mid hip and neck
            neck = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
                    (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2]
            
            mid_hip = [(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
                       (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2]


            # Calculate the angle of the spine with respect to the vertical axis
            spine_angle = calculate_angle(mid_hip, neck, [neck[0], neck[1] - 1])

            # Calculate angles
            angle = calculate_angle(right_shoulder, elbow, wrist)

            angle_knee = calculate_angle(hip, knee, ankle) 
            angle_knee = round(angle_knee, 2)

            angle_hip = calculate_angle(right_shoulder, hip, knee)
            angle_hip = round(angle_hip, 2)

            hip_angle = 180 - angle_hip
            knee_angle = 180 - angle_knee

            angle_min.append(angle_knee)
            angle_min_hip.append(angle_hip)

            vertical_point = [right_shoulder[0], right_shoulder[1] - 1]
            shoulder_angle = calculate_angle(left_shoulder, right_shoulder, vertical_point)

            cv2.putText(
                image,
                str(angle_knee),
                tuple(np.multiply(knee, [1500, 800]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                str(angle_hip),
                tuple(np.multiply(hip, [1500, 800]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            print(spine_angle)
            if spine_angle < 170:
                cv2.putText(
                    image,
                    "Bend down a bit",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif spine_angle > 180:
                cv2.putText(
                    image,
                    "Stand up a bit",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )


            # Check if user is standing straight
            if shoulder_angle < 86:
                cv2.putText(
                    image,
                    "Turn a bit to your left",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if shoulder_angle > 92:
                cv2.putText(
                    image,
                    "Turn a bit to your right",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            

            if angle_knee > 159:
                stage = "up"
            if angle_knee <= 90 and stage == "up":
                stage = "down"
                counter += 1
                min_ang = min(angle_min)
                max_ang = max(angle_min)

                min_ang_hip = min(angle_min_hip)
                max_ang_hip = max(angle_min_hip)

                angle_min = []
                angle_min_hip = []

            # Let user go a little deeper, if they didn't go all down
            if stage == "down" and (angle_hip >= 60 or angle_knee >= 90):
                cv2.putText(
                    image,
                    "Go deeper!!",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        except Exception as e:
            pass

        # Get the size of the video frame dynamically
        frame_height, frame_width = image.shape[:2]

        # Define the size and position of the rectangle
        rect_width, rect_height = 220, 130
        rect_x = frame_width - rect_width - 20  
        rect_y = frame_height - rect_height - 20 

        # Draw the rectangle
        cv2.rectangle(
            image,
            (rect_x, rect_y),
            (rect_x + rect_width, rect_y + rect_height),
            (0, 0, 0),
            -1,
        )

        # Rep data
        cv2.putText(
            image,
            "Squats : " + str(counter),
            (rect_x + 10, rect_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Knee angle
        cv2.putText(
            image,
            "Knee-joint angle : " + str(min_ang),
            (rect_x + 10, rect_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Hip angle
        cv2.putText(
            image,
            "Hip-joint angle : " + str(min_ang_hip),
            (rect_x + 10, rect_y + 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=2, circle_radius=2),
        )

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
