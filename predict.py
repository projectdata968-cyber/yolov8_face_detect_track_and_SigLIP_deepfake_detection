import os
import cv2
import argparse
from detector import FaceDetector, SigLIPDetector
import time

# Modified "E:\face_tracking\yolov8-face\ultralytics\nn\tasks.py", function "torch_safe_load" updated.

def run(video_path, yolo_weights, device, conf_threshold):
    cap = cv2.VideoCapture(video_path)

    face_detector = FaceDetector(yolo_weights, device)
    deepfake_detector = SigLIPDetector(device, conf_threshold)

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect + Track
            faces = face_detector.detect_and_track(frame)

            for track_id, x, y, w, h in faces:
                face = face_detector.extract_face(frame, x, y, w, h)

                result = deepfake_detector.predict(face)

                if result is None:
                    continue  # skip low confidence

                label, conf = result

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = f"ID:{track_id} {label} ({conf:.2f})"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Deepfake Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting cleanly...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--weights", type=str, default=os.path.join("weights", "yolov8n-face.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)

    args = parser.parse_args()

    run(args.video, args.weights, args.device, args.conf)