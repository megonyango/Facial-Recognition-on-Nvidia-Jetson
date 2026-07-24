"""
Facial Recognition on Nvidia Jetson - Logitech USB Camera
=========================================================
SSD DNN face detector (CUDA-accelerated) + LBPH recognizer.

SETUP (run once on the board)
-----------------------------
  sudo apt update
  sudo apt install -y python3-opencv v4l-utils
  pip3 install opencv-contrib-python numpy        # only if cv2.face is missing

  wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
  wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel

CHECK YOUR CAMERA FIRST
-----------------------
  ls /dev/video*                                  # confirm the index
  v4l2-ctl --list-formats-ext -d /dev/video0      # confirm MJPG + resolutions

USAGE
-----
  python3 jetson_face_recognition.py --enroll "Meg"
  python3 jetson_face_recognition.py --train
  python3 jetson_face_recognition.py --run

Press 'q' in any window to quit.
"""

import argparse
import os
import pickle
import time

import cv2
import numpy as np

# ============================================================
# CONFIGURATION - tune these
# ============================================================
PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

DATA_DIR        = "faces"            # one subfolder per enrolled person
RECOGNIZER_FILE = "recognizer.yml"
LABELS_FILE     = "labels.pkl"

CAMERA_INDEX    = 0                  # change if /dev/video0 isn't your Logitech
FRAME_W         = 1280
FRAME_H         = 720
FPS             = 30
LOCK_AUTOFOCUS  = True               # True stops C920/Brio focus hunting indoors

CONF_THRESHOLD  = 0.6                # min detector confidence (0-1)
LBPH_THRESHOLD  = 70                 # max distance to accept a match (lower = stricter)
FACE_SIZE       = (200, 200)         # all faces normalised to this
ENROLL_SAMPLES  = 30                 # frames captured per person
ENROLL_DELAY    = 0.15               # seconds between captures - lets you shift pose


# ============================================================
# CAMERA
# ============================================================
def open_camera():
    """
    Open the Logitech over V4L2.

    The MJPG line is the important one: most Logitech cams default to raw
    YUYV, which saturates USB bandwidth and caps you around 5-10 FPS at 720p.
    Forcing MJPG gets the full 30.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)      # drop stale frames, cuts visible lag

    if LOCK_AUTOFOCUS:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {CAMERA_INDEX}. "
            "Run 'ls /dev/video*' and update CAMERA_INDEX."
        )

    # Report what we actually got - the camera may refuse a requested mode
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera open at {w}x{h}")
    return cap


# ============================================================
# DETECTOR
# ============================================================
def load_detector():
    """Load the SSD face detector and push it onto the GPU if possible."""
    if not (os.path.exists(PROTO) and os.path.exists(MODEL)):
        raise FileNotFoundError(
            "Detector model files missing. See the wget commands in the header."
        )

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

    # JetPack's bundled OpenCV is normally built with CUDA. If yours isn't,
    # this falls back to CPU - you'll see the warning below.
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        # Force a dummy pass so a missing CUDA build fails here, not mid-loop
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        net.setInput(cv2.dnn.blobFromImage(dummy, 1.0, (300, 300),
                                           (104.0, 177.0, 123.0)))
        net.forward()
        print("[INFO] CUDA backend active (FP16)")
    except cv2.error:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[WARN] OpenCV not built with CUDA - running on CPU. Expect lower FPS.")

    return net


def detect_faces(net, frame):
    """Return face boxes as (x1, y1, x2, y2), largest first."""
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    # Largest face first - stops a bystander in the background being enrolled
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def crop_face(frame, box):
    """Grey, resize and equalise a face crop so all samples are comparable."""
    x1, y1, x2, y2 = box
    face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, FACE_SIZE)
    return cv2.equalizeHist(face)      # evens out lighting between rooms


# ============================================================
# ENROLLMENT
# ============================================================
def enroll(name):
    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.endswith(".png")])
    net = load_detector()
    cap = open_camera()

    print(f"[INFO] Enrolling '{name}' - {ENROLL_SAMPLES} samples.")
    print("[INFO] Vary your angle, expression and distance as it captures.")

    count = 0
    last_capture = 0.0
    try:
        while count < ENROLL_SAMPLES:
            ok, frame = cap.read()
            if not ok:
                continue

            boxes = detect_faces(net, frame)

            if boxes and (time.time() - last_capture) > ENROLL_DELAY:
                box = boxes[0]                          # largest face only
                face = crop_face(frame, box)
                path = os.path.join(person_dir, f"{existing + count:03d}.png")
                cv2.imwrite(path, face)
                count += 1
                last_capture = time.time()
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"{name}: {count}/{ENROLL_SAMPLES}", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Enrollment", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Captured {count} samples for '{name}'.")
    train()


# ============================================================
# TRAINING
# ============================================================
def train():
    faces, ids, label_map = [], [], {}

    if not os.path.isdir(DATA_DIR):
        print("[ERROR] No faces/ directory. Run --enroll first.")
        return None, None

    for idx, person in enumerate(sorted(os.listdir(DATA_DIR))):
        pdir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(pdir):
            continue
        label_map[idx] = person
        for f in sorted(os.listdir(pdir)):
            img = cv2.imread(os.path.join(pdir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                ids.append(idx)

    if not faces:
        print("[ERROR] No enrolled images found.")
        return None, None

    if len(label_map) < 2:
        print("[WARN] Only one person enrolled - LBPH will match everyone to them.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save(RECOGNIZER_FILE)

    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[INFO] Trained on {len(faces)} images, {len(label_map)} people.")
    print(f"[INFO] Labels: {label_map}")
    return recognizer, label_map


# ============================================================
# LIVE RECOGNITION
# ============================================================
def run():
    if not (os.path.exists(RECOGNIZER_FILE) and os.path.exists(LABELS_FILE)):
        print("[ERROR] No trained model. Run --enroll first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_FILE)
    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)

    net = load_detector()
    cap = open_camera()

    print("[INFO] Running. Press 'q' to quit.")
    frames, t0, fps = 0, time.time(), 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            for box in detect_faces(net, frame):
                face = crop_face(frame, box)
                label_id, distance = recognizer.predict(face)

                # LBPH always returns its nearest match, even for a stranger.
                # LBPH_THRESHOLD is the only thing separating known from unknown.
                if distance < LBPH_THRESHOLD:
                    name, color = label_map.get(label_id, "Unknown"), (0, 255, 0)
                else:
                    name, color = "Unknown", (0, 0, 255)

                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({distance:.0f})",
                            (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            frames += 1
            if frames % 30 == 0:
                fps = 30 / (time.time() - t0)
                t0 = time.time()
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Jetson Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jetson face recognition demo")
    ap.add_argument("--enroll", metavar="NAME", help="Enroll a person by name")
    ap.add_argument("--train", action="store_true", help="Retrain from stored images")
    ap.add_argument("--run", action="store_true", help="Run live recognition")
    args = ap.parse_args()

    if args.enroll:
        enroll(args.enroll)
    elif args.train:
        train()
    elif args.run:
        run()
    else:
        ap.print_help()
