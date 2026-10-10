# ASL Static Sign Recognition

A webcam-based application that detects a hand with MediaPipe, converts the
hand landmarks into features, and (in later milestones) classifies static ASL
alphabet signs with a scikit-learn model trained on a self-collected dataset.

## What this project is, and is not

- **Is:** recognition of **static** fingerspelled letters from a single hand
  pose in a single frame.
- **Is not:** sign-language translation. It does not recognize words, phrases,
  grammar, facial expressions or movement. Signing "hello" or "thank you"
  will not produce meaningful output.
- **Letters covered:** 24 letters (A-Y, without J). J and Z are excluded because
  they are signed with motion and cannot be captured in one still frame.
- **One hand only**, facing the camera, with reasonable lighting.

## How it works

```
Webcam frame -> MediaPipe hand landmarks (21 points x 3 coordinates)
             -> normalize (wrist as origin, scale by hand size) -> 63 features
             -> classifier -> confidence threshold + smoothing -> predicted letter
```

Normalization means the same hand shape gives similar features wherever the
hand is in the frame and however far it is from the camera.

## Project status

| Milestone | Description | Status |
|---|---|---|
| 1 | Project foundation, hand detection, landmark features, tests | Code added |
| 2 | Dataset collection tool | Planned |
| 3 | Training, validation, held-out test evaluation, confusion matrix | Planned |
| 4 | Real-time prediction, confidence threshold, smoothing | Planned |
| 5 | Streamlit interface | Planned |
| 6 | Final documentation and polish | Planned |

Evaluation results will be added here once a model has been trained and
evaluated on a dataset collected for this project.

## Setup (Windows PowerShell)

Python 3.10, 3.11 or 3.12 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run the tests

```powershell
pytest
```

## Check your webcam and hand detection

```powershell
python scripts/check_webcam.py
```

Press `q` in the window to quit. Use `--camera-index 1` to try another camera.

## Project structure

```
src/asl_recognition/   core code (config, logging, hand detection, features)
scripts/               command-line tools
app/                   Streamlit app (later milestone)
tests/                 automated tests
data/                  collected datasets (not committed by default)
models/                trained models (not committed by default)
reports/               evaluation outputs
```

## Privacy

Everything runs locally. No video or landmark data is sent anywhere, and no
paid or cloud services are used.