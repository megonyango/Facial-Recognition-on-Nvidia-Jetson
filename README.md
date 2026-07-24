# Face Detection & Recognition on the NVIDIA Jetson Nano

A small, well-commented OpenCV pipeline for running face **detection** and **recognition** on an NVIDIA Jetson Nano with a USB webcam. Built as a hands-on lab exercise, so the code favours clarity over cleverness — it's meant to be read, run, and modified.

It also documents an unexpected finding that turned out to be the most interesting part of the whole exercise: **the camera, not the model, was the thing that failed** — and it failed hardest on dark-skinned faces. See [The finding that matters](#the-finding-that-matters) below.

---

## Contents

- [What it does](#what-it-does)
- [Hardware & software](#hardware--software)
- [Install](#install)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [The finding that matters](#the-finding-that-matters)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Credits & licence](#credits--licence)

---

## What it does

1. **Detects** faces in a live camera feed using a pre-trained deep-learning detector (ResNet-10 SSD).
2. **Enrols** people by capturing sample images of each face.
3. **Recognises** enrolled people in real time, labelling each face with a name (or `Unknown`).

Everything runs from one script with three modes: `--enroll`, `--train`, `--run`.

---

## Hardware & software

| | |
|---|---|
| **Board** | NVIDIA Jetson Nano (tested on the 2GB Developer Kit) |
| **OS image** | JetPack 4.6.1 / L4T R32.7.1 (Ubuntu 18.04) |
| **Camera** | Any USB webcam (developed with a Logitech UHD cam) |
| **Language** | Python 3 |
| **Key library** | OpenCV **with contrib modules** (the `cv2.face` module is required) |

> **Note on the GPU:** the script includes a CUDA acceleration path, but it falls back to the CPU automatically if CUDA isn't available. On CPU-only it still runs — just slower. See [Known limitations](#known-limitations).

---

## Install

**1. System packages**

```bash
sudo apt update
sudo apt install -y python3-opencv v4l-utils
```

**2. Python packages** (only needed if `cv2.face` is missing)

```bash
pip3 install opencv-contrib-python numpy
```

Check it worked:

```bash
python3 -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'face'))"
# The second line must print True
```

**3. Download the detector model** (two files, one-time)

```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

Keep both files in the same folder as the script.

**4. Check your camera**

```bash
ls /dev/video*                                # confirm the device index
v4l2-ctl --list-formats-ext -d /dev/video0    # confirm it supports MJPG
```

If your camera isn't at index `0`, update `CAMERA_INDEX` in the script.

---

## Usage

The workflow is always **enrol → (auto-train) → run**.

**Enrol each person** (captures 30 samples, then trains automatically):

```bash
python3 jetson_face_recognition.py --enroll "Alice"
python3 jetson_face_recognition.py --enroll "Brian"
```

> ⚠️ **Enrol at least two people.** With only one person enrolled, the recogniser labels *everyone* as that person. This is a property of the method, not a bug — see [How it works](#how-it-works).

During enrolment, slowly vary your angle, expression, and distance so the samples aren't 30 near-identical frames.

**Re-train manually** (e.g. after adding images by hand):

```bash
python3 jetson_face_recognition.py --train
```

**Run live recognition:**

```bash
python3 jetson_face_recognition.py --run
```

Press **`q`** in any window to quit.

---

## How it works

The pipeline separates **detection** (finding faces) from **recognition** (naming them). They're kept separate on purpose — they fail in different ways, so separating them makes each failure easy to see.

```
Camera frame
     │
     ▼
[1] DETECT  ─ ResNet-10 SSD ─ finds face boxes, keeps the largest first
     │
     ▼
[2] NORMALISE ─ crop → greyscale → resize 200×200 → equalise brightness
     │
     ▼
[3] RECOGNISE ─ LBPH ─ nearest-neighbour match against enrolled faces
     │
     ▼
Label + box drawn on the frame
```

**Detector — ResNet-10 SSD.** Chosen over the classic Haar cascade because Haar degrades badly on tilted heads and on faces with low contrast against the background. The SSD is more robust and still light enough for the Nano.

**Recogniser — LBPH (Local Binary Pattern Histograms).** A classic, non-learned method. It's fast, trains in seconds, and runs in a few milliseconds per face on CPU — a deliberate choice for weak hardware. Two things to understand about it:

- It returns a **distance**, not a probability. **Lower = better match.**
- It **always returns the nearest enrolled identity**. Shown a stranger, it won't say "unknown" on its own — it names the closest person it knows. The `LBPH_THRESHOLD` cut-off is the *only* thing separating a real match from a false one, so calibrating it matters.

**A practical tip on tuning the threshold:** run `--run`, stand in front of the camera yourself, and note the distance printed next to your name (usually ~30–55 for a real match). Then have someone *not* enrolled stand in and note theirs (usually 80+). Set `LBPH_THRESHOLD` in the gap between the two.

---

## Configuration

All tunable settings live at the top of the script:

| Setting | Default | What it does |
|---|---|---|
| `CAMERA_INDEX` | `0` | Which `/dev/videoN` to open |
| `FRAME_W`, `FRAME_H` | `1280×720` | Capture resolution. Drop to `640×480` for more speed |
| `LOCK_AUTOFOCUS` | `True` | Stops the camera focus-hunting indoors |
| `CONF_THRESHOLD` | `0.6` | Minimum detector confidence to accept a face |
| `LBPH_THRESHOLD` | `70` | Max distance to accept a match — **lower = stricter** |
| `ENROLL_SAMPLES` | `30` | How many samples to capture per person |

**Speed tip:** if the frame rate is low, lower the capture resolution first. The detector shrinks every frame to 300×300 internally anyway, so you lose almost no accuracy but save a lot of transfer and conversion cost.

---

## The finding that matters

While testing, we ran the detector on a group photo the rig captured — **8 people**, all facing the camera in a normal classroom.

**The detector found 0 of the 8 faces.**

It looked like the model was broken. But when we simply **brightened the image** and ran the *exact same detector with the exact same settings*, it found **4 of the 8** — confidently.

| Version of the photo | Faces found (of 8) |
|---|---|
| As the camera saved it | **0** |
| After brightening (histogram equalisation) | **4** |
| After brightening (gamma correction) | **4** |
| After milder brightening (CLAHE) | 2 |

*Same detector, same settings every time. Only the brightness changed.*

**What happened:** the photo was severely underexposed (average brightness 51/255; ~40% of pixels near-black). There was a bright wall display and ceiling light in the shot, so the camera's **automatic exposure** dimmed everything to compensate — pushing the actual people into darkness where the detector had nothing to work with.

**Why it matters:**

- **The model did nothing wrong.** Its ability was there the whole time — it just needed a usable image.
- **The failure happened *before* any AI ran.** It was created by the camera's exposure setting, not the model, its training, or its design.
- **It was invisible.** The camera reported success and saved a normal-looking file. Nothing flagged a problem.
- **It fell hardest on dark-skinned faces** — matching well-documented research ([Buolamwini & Gebru, 2018](http://proceedings.mlr.press/v81/buolamwini18a.html); [NIST FRVT Part 3, 2019](https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.8280.pdf)).

**The takeaway:** getting exposure and lighting right at the camera is not optional preprocessing — it's part of making the system work at all, and work *fairly*. Don't trust automatic exposure. Set it by hand and check each frame's brightness.

> This is why the roadmap below puts **manual exposure control** and a **brightness check** at the very top. The current code does **not** yet include them.

---

## Known limitations

- **GPU may be unavailable.** On the board this was developed on, the GPU could not be initialised (a hardware fault), so everything ran on CPU. The script handles this gracefully but runs slower.
- **LBPH is basic.** It works well only for cooperative subjects facing the camera under steady light. It tolerates only ~15–20° of head turn.
- **It can't reject strangers on its own.** The threshold is the only defence against false matches.
- **Small / distant faces are lost.** The detector rescales every frame to 300×300, so faces far from the camera become too small to find.
- **No liveness detection.** A printed photo will fool it. **Not suitable for security or authentication as-is.**
- **No exposure control yet.** See [the finding above](#the-finding-that-matters).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `hasattr(cv2, 'face')` prints `False` | Install `opencv-contrib-python`, then restart the shell/session |
| `Could not open camera at index 0` | Run `ls /dev/video*` and update `CAMERA_INDEX` |
| Camera works then vanishes after replug | Device indices aren't stable; resolve by path under `/dev/v4l/by-id/` |
| Very low frame rate (~5 fps) | Camera is sending raw YUYV; the script forces MJPG, but also try lowering resolution |
| Everyone is labelled as one person | You enrolled only one person — enrol at least two |
| Faces not detected in dim rooms | **Read [The finding that matters](#the-finding-that-matters).** Fix your exposure/lighting |

---

## Roadmap

Planned improvements, in priority order:

- [ ] **Manual exposure control** — disable auto-exposure, set it for the subjects (`v4l2-ctl --set-ctrl=exposure_auto=1` then `exposure_absolute=<value>`)
- [ ] **Brightness check at capture** — measure mean frame luminance and warn when a frame is too dark to use
- [ ] **Photometric normalisation** — brighten each frame before detection
- [ ] **Resolve camera by persistent path** instead of numeric index
- [ ] **Access logging** — timestamped name + confidence + snapshot to CSV, for attendance-style use
- [ ] **Stronger recogniser** — swap LBPH for a learned embedding (e.g. MobileFaceNet) once GPU compute is available

Contributions welcome — open an issue or a pull request.

---

## Credits & licence

- Face detector: [OpenCV DNN samples](https://github.com/opencv/opencv/tree/master/samples/dnn) (ResNet-10 SSD, pre-trained).
- Recogniser: LBPH from the [OpenCV contrib](https://github.com/opencv/opencv_contrib) `face` module.

Built as a laboratory exercise by the Doctoral Programme in Computer Science, Strathmore University (Cohort III).

Released under the **MIT Licence** — free to use, modify, and share. See `LICENSE`.
