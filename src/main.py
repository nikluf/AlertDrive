import cv2
import time
import numpy as np
import argparse
import sys

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]  
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
OUTER_LIP_IDX = [61, 291, 0, 17, 78, 308, 13]  

def euclid(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_idx):
    
    p = [landmarks[i] for i in eye_idx]
    
    A = euclid(p[0], p[3])
   
    B = euclid(p[1], p[5])
    C = euclid(p[2], p[4])
    ear = (B + C) / (2.0 * A + 1e-6)
    return ear

def mouth_aspect_ratio(landmarks):
    
    left = landmarks[61]
    right = landmarks[291]
    top = landmarks[13]
    bottom = landmarks[14] if 14 < len(landmarks) else landmarks[17]
    A = euclid(top, bottom)
    B = euclid(left, right)
    mar = A / (B + 1e-6)
    return mar

def head_pose_proxy(landmarks):
    
    nose = landmarks[1]
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    mid_eye = ((left_eye_outer[0] + right_eye_outer[0]) / 2.0, (left_eye_outer[1] + right_eye_outer[1]) / 2.0)

    
    yaw = nose[0] - mid_eye[0]
    
    pitch = nose[1] - mid_eye[1]
    return yaw, pitch

def beep():
  
    print("\a", end="")

def put_label(frame, text, org, scale=0.7, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (48,48,48), thickness, cv2.LINE_AA)

def norm_landmarks(landmark_list, w, h):
    return [(lm.x, lm.y) for lm in landmark_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--ear_thresh", type=float, default=0.21)
    ap.add_argument("--ear_frames", type=int, default=12)
    ap.add_argument("--mar_thresh", type=float, default=0.60)
    ap.add_argument("--mar_frames", type=int, default=15)
    ap.add_argument("--yaw_thresh", type=float, default=0.12)
    ap.add_argument("--pitch_thresh", type=float, default=0.10)
    ap.add_argument("--look_frames", type=int, default=15)
    args = ap.parse_args()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("ERROR: Could not open camera", args.cam)
        sys.exit(1)

    ear_counter = 0
    mar_counter = 0
    look_counter = 0
    last_alert = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = face_mesh.process(rgb)
        alert_msgs = []

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = norm_landmarks(lm, w, h)

            #EAR
            left_ear = eye_aspect_ratio(pts, LEFT_EYE_IDX)
            right_ear = eye_aspect_ratio(pts, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0

            #MAR
            mar = mouth_aspect_ratio(pts)

            #Head pose proxy
            yaw, pitch = head_pose_proxy(pts)

            #Updating counters
            if ear < args.ear_thresh:
                ear_counter += 1
            else:
                ear_counter = 0

            if mar > args.mar_thresh:
                mar_counter += 1
            else:
                mar_counter = 0

            if abs(yaw) > args.yaw_thresh or pitch > args.pitch_thresh:
                look_counter += 1
            else:
                look_counter = 0

            #Alerts
            if ear_counter >= args.ear_frames:
                alert_msgs.append("DROWSY: Eyes closed")
            if mar_counter >= args.mar_frames:
                alert_msgs.append("DROWSY: Yawning")
            if look_counter >= args.look_frames:
                alert_msgs.append("DISTRACTED: Look ahead")

           
            now = time.time()
            if alert_msgs and (now - last_alert) > 1.5:
                beep()
                last_alert = now

            #HUD
            put_label(frame, f"EAR: {ear:.2f}", (10, 30))
            put_label(frame, f"MAR: {mar:.2f}", (10, 60))
            put_label(frame, f"Yaw: {yaw:.2f}  Pitch: {pitch:.2f}", (10, 90))
            y0 = 130
            for m in alert_msgs:
                put_label(frame, m, (10, y0), scale=0.8, thickness=2)
                y0 += 30
        else:
            
            put_label(frame, "No face detected", (10, 30))

        cv2.imshow("Driver Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
