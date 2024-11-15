# Frame Interpolation
By: Ethan Kostiuk & Mustahid Rafi

## About
Implementing an algorithm from the paper:
"Yahia, H. B. & Reisser, M. (2016). Frame interpolation using convolutional neural networks on
2d animation. Bachelor Thesis, XP055558906, 1-19."

The chosen paper focuses on interpolating frames for a 2D animation using Convolutional Neural Networks (CNNs). Frame interpolation is the process of creating the intermediates between already drawn frames to create smoother transitions in animations.

## How to run
Prerequisites:
- Python 3.9-3.12
- pip version >= 19.0
- NVIDIA GPU drvivers
  - >= 525.60.13 for Linux
  - >= 528.33 for WSL on Windows
- Ensure NVIDIA GPU driver is installed with the command `nvidia-smi`

1. Create a virtual environment:
`python3 -m venv tf`

2. Activate the virtual environment:
`source tf/bin/actiavate`

3. Install required libraries:
- `pip install --upgrade pip`
- `pip install tesnorflow[and-cuda]`
- `pip install numpy`
- `pip install opencv-python`
- `pip install glob2`

4. Run
`python3 [nameOfFile.py]`

## Sites used
https://animeclips.online/home/
https://ezgif.com/video-to-jpg

https://y2meta.tube/convert/?videoId=HZulBIgRLLE
https://www.youtube.com/watch?v=HZulBIgRLLE

## Training
