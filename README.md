# Computer Vision Assignment 1 – CameraApp

A Python OpenCV application for real-time webcam capture, image processing, panorama creation, camera calibration, and augmented reality.  
This project fulfills **Assignment #1** requirements.

---

## ✅ Features Implemented

- **Image Processing**: RGB ↔ Gray ↔ HSV, contrast/brightness, histogram  
- **Filters**: Gaussian, Bilateral  
- **Edges & Lines**: Canny, Hough transform  
- **Transformations**: Translation, rotation, scaling  
- **Panorama**: Manual stitching (ORB + Homography, not built-in OpenCV function)  
- **Camera Calibration**: With chessboard pattern (`A4_Chessboard_9x6.png`)  
- **Augmented Reality**: AR marker detection + T-Rex model overlay  

---

## ⚙️ Environment Setup

### 1. Using [uv](https://github.com/astral-sh/uv) (recommended)

```bash
# Create and activate venv
uv venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Sync dependencies
uv sync
2. Using pip (fallback)
If uv is not available, install dependencies directly:

bash
Copy code
pip install -r requirements.txt
requirements.txt should contain:

shell
Copy code
numpy>=2.2.6
opencv-python>=4.12.0.88
opencv-contrib-python>=4.12.0.88
▶️ Running the App
Using uv:

bash
Copy code
uv run main.py
Using pip:

bash
Copy code
python main.py
🎮 Controls
Key	Function
1	Color mode
2	Grayscale mode
3	HSV mode
A	Brightness/Contrast
H	Histogram
G	Gaussian blur
B	Bilateral filter
C	Canny edges
D	Hough lines
T	Transform mode
0	Reset transform
P	Panorama mode ON/OFF
Z	Capture panorama frame
O	Build panorama
X	Reset panorama
K	Calibrate camera
R	Run AR mode
Q	Quit

📷 Calibration & AR
Calibration
Print A4_Chessboard_9x6.png

Press K to start calibration (needs 15 captures)

Produces calibration.npz

Augmented Reality
Print A4_ArUco_Marker.png

Press R to start AR mode

Projects trex_model.obj on the detected marker

📝 Notes
Requires Python >= 3.12

Needs a working webcam

Outputs are saved under the output/ folder