# CameraApp

A simple Python OpenCV application for real-time webcam capture with interactive color modes, brightness/contrast adjustment, histogram visualization, and Gaussian blur.

## Features

- **Live Webcam Feed**  
    Displays real-time video from your default camera.

- **Color Modes**  
    - `COLOR`: Standard BGR color.
    - `GRAY`: Grayscale.
    - `HSV`: HSV color space.

- **Brightness & Contrast Adjustment**  
    - Toggle adjustment mode with `B`.
    - Adjust contrast (`Alpha`) and brightness (`Beta`) using trackbars.

- **Histogram Visualization**  
    - Toggle histogram display with `S`.
    - Shows intensity distribution for the current mode.

- **Gaussian Blur**  
    - Toggle Gaussian blur with `G`.
    - Smooths the video feed for noise reduction.

- **Bilateral Filter**  
    - Toggle Gaussian blur with `B`.
    - Applies edge-preserving smoothing to reduce noise while keeping edges sharp.

- **Canny Edge Detection**
    - Toggle Canny edge detection with `C`.
    - Detects edges in the video feed using adjustable thresholds.

- **Hough Line Detection**
    - Toggle Hough line detection with `D`.

- **Keyboard Controls**
    - `1`: Switch to Color mode
    - `2`: Switch to Grayscale mode
    - `3`: Switch to HSV mode
    - `A`: Toggle brightness/contrast adjustment
    - `H`: Toggle histogram display
    - `G`: Toggle Gaussian blur
    - `B`: Toggle Bilateral filter
    - `Q`: Quit the application

## Usage

1. **Install [uv](https://github.com/astral-sh/uv):**  
   Follow the instructions on the [uv GitHub page](https://github.com/astral-sh/uv) to install `uv` for your platform.

2. **Set up the project:**
    ```bash
    # Navigate to your project directory
    cd path/to/your/project

    # Create a new virtual environment
    uv venv .venv

    # Activate the virtual environment
    # On Unix/macOS:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate

    # Sync dependencies (ensure you have a requirements.txt or pyproject.toml)
    uv sync
    ```

3. **Run the app:**
    ```bash
    uv run main.py
    ```

4. **Interact using the keyboard shortcuts above.**

## Notes

- Requires a working webcam.
- Tested with Python 3.x and OpenCV 4.x.

## License

AIT License.