# Hard Hat Detection

A YOLO11-based object detector that identifies whether people in an image or video
are wearing a hard hat. Trained on the [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t)
dataset (version 17) with two classes:

- `helmet` — wearing a hard hat (compliant)
- `head` — bare head (not compliant)

## Setup

```bash
python -m venv yolo-env
# Windows
yolo-env\Scripts\activate
# macOS / Linux
source yolo-env/bin/activate

pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled build of PyTorch matching your hardware
(see https://pytorch.org/get-started/locally/).

## Usage

### 1. Download the dataset

The Roboflow API key is read from an environment variable, not hardcoded.

```bash
# Windows (PowerShell)
$env:ROBOFLOW_API_KEY = "your_key_here"
# macOS / Linux
export ROBOFLOW_API_KEY="your_key_here"

python get_data.py
```

This downloads the dataset into `Hard-Hat-Universe-17/`.

### 2. Train

```bash
python train.py
```

Trains YOLO11s for 50 epochs at 640px. Weights are saved to
`runs/detect/train/weights/best.pt`.

### 3. Run live detection

```bash
python live.py
```

Loads the trained `best.pt` and runs detection on the webcam feed.

## Notes

The dataset, virtual environment, and trained weights are not tracked in git
(see `.gitignore`) — they are large and regenerable.
