import cv2
import numpy as np
import os

class CameraApp:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.mode = "COLOR"  # Modes: COLOR, GRAY, HSV
        self.alpha = 1.0  # Contrast control
        self.beta = 0     # Brightness control

        # NEW: calibration + AR
        self.camera_matrix, self.dist = self.load_calibration()
        self.obj_vertices, self.obj_faces = self.load_obj("trex_model.obj")

        # Panorama
        self.panorama_frames = []
        self.panorama_status = ""
        self.status_counter = 180


        # Flags
        self.show_hist = False
        self.show_adjust = False
        self.show_gaussian = False
        self.show_bilateral = False
        self.show_canny = False
        self.hough_transform = False
        self.transform_mode = False   # NEW: Transform mode
        self.panorama_mode = False


        # State
        self.active_trackbar_mode = None

        cv2.namedWindow('Camera')

    # ---------- UTILS ----------
    def nothing(self, x): pass

    def remove_trackbars(self):
        cv2.destroyWindow('Camera')
        cv2.namedWindow('Camera')
        self.active_trackbar_mode = None

    # ---------- TRACKBARS ----------
    def create_trackbars_hough_lines(self):
        cv2.createTrackbar('Threshold', 'Camera', 50, 200, self.nothing)
        cv2.createTrackbar('Min Line Length', 'Camera', 50, 200, self.nothing)
        cv2.createTrackbar('Max Line Gap', 'Camera', 10, 100, self.nothing)

    def create_trackbars_canny(self):
        cv2.createTrackbar('Threshold1', 'Camera', 50, 500, self.nothing)
        cv2.createTrackbar('Threshold2', 'Camera', 150, 500, self.nothing)

    def create_trackbars_adjust(self):
        cv2.createTrackbar('Alpha x0.1', 'Camera', int(self.alpha*10), 30, self.nothing)
        cv2.createTrackbar('Beta', 'Camera', self.beta+100, 200, self.nothing)

    def create_trackbars_gaussian(self):
        cv2.createTrackbar('Kernel Size', 'Camera', 1, 20, self.nothing)
        cv2.createTrackbar('SigmaX', 'Camera', 0, 100, self.nothing)

    def create_trackbars_bilateral(self):
        cv2.createTrackbar('Diameter', 'Camera', 1, 20, self.nothing)
        cv2.createTrackbar('SigmaColor', 'Camera', 0, 100, self.nothing)
        cv2.createTrackbar('SigmaSpace', 'Camera', 0, 100, self.nothing)

    def create_trackbars_transform(self):
        cv2.createTrackbar('Translate X', 'Camera', 50, 100, self.nothing)
        cv2.createTrackbar('Translate Y', 'Camera', 50, 100, self.nothing)
        cv2.createTrackbar('Rotation', 'Camera', 0, 360, self.nothing)
        cv2.createTrackbar('Scale x10', 'Camera', 10, 30, self.nothing)

    # ---------- HISTOGRAM ----------
    def show_histogram(self, frame):
        if frame is None:
            return np.zeros((300, 512, 3), dtype=np.uint8)
        h, w = 300, 512
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        num_channels = 1 if self.mode == "GRAY" else 3

        if self.mode == "GRAY":
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for i in range(num_channels):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, h, cv2.NORM_MINMAX)
            color = colors[i] if num_channels == 3 else (255, 255, 255)
            for x in range(1, 256):
                cv2.line(hist_img,
                         (x - 1, h - int(hist[x - 1])),
                         (x, h - int(hist[x])),
                         color, 2)
        return hist_img

    # ---------- IMAGE MODES ----------
    def process_color(self, frame): return frame

    def process_gray(self, frame):
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return frame

    def process_hsv(self, frame):
        if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ---------- HELP TEXT ----------
    def draw_help_text(self, display_frame):
        if self.panorama_mode:
            help_text = "Panorama Mode | Z: Capture | O: Build | X: Reset | P: Exit Panorama"
        else:
            help_text = ("1: Color | 2: Gray | 3: HSV | A: Adjust | H: Histogram | "
                        "G: Gaussian | B: Bilateral | C: Canny | D: Hough | "
                        "K: Calibrate | R: AR Mode | T: Transform | 0: Reset Transform | "
                        "P: Panorama")

        quit_text = "Q: Quit"
        mode_text = f"Mode: {self.mode}"
        lines = []

        frame_width = display_frame.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2


        def split_text(text, max_width):
            words = text.split(' ')
            lines = []
            current = ""
            for word in words:
                test = current + (' ' if current else '') + word
                size = cv2.getTextSize(test, font, font_scale, thickness)[0][0]
                if size > max_width and current:
                    lines.append(current)
                    current = word
                else:
                    current = test
            if current:
                lines.append(current)
            return lines

        max_text_width = int(frame_width * 0.95)
        lines.extend(split_text(help_text, max_text_width))
        lines.append(quit_text)
        lines.append(mode_text)
        if self.show_adjust:
            lines.append(f"Alpha: {self.alpha:.1f}  Beta: {self.beta}")

        y = 30
        for i, text in enumerate(lines):
            color = (113, 179, 60)
            if i == len(lines) - 1: color = (255, 0, 0)
            elif i == len(lines) - 2: color = (0, 0, 255)
            cv2.putText(display_frame, text, (10, y), font, font_scale, color, thickness)
            y += 25

    # ---------- CALIBRATION + OBJ ----------
    def load_calibration(self):
        if os.path.exists("calibration.npz"):
            data = np.load("calibration.npz")
            print("Loaded calibration.npz")
            return data["mtx"], data["dist"]
        return None, None

    def calibrate_camera(self, chessboard_size=(9, 6)):
        print("Starting camera calibration... show chessboard.")
        objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
        objpoints, imgpoints = [], []
        good = 0

        while good < 15:
            ret, frame = self.cam.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if found:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(frame, chessboard_size, corners2, found)
                good += 1
                print(f"Captured {good}/15")
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        cv2.destroyWindow("Calibration")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez("calibration.npz", mtx=mtx, dist=dist)
        self.camera_matrix, self.dist = mtx, dist
        print("Calibration complete! Saved calibration.npz")

    def load_obj(self, filename):
        if not os.path.exists(filename): return np.array([]), []
        verts, faces = [], []
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append(list(map(float, line.strip().split()[1:4])))
                elif line.startswith("f"):
                    face = [int(idx.split("/")[0]) - 1 for idx in line.strip().split()[1:]]
                    faces.append(face)
        return np.array(verts, dtype=np.float32), faces

    # ---------- AR DRAW ----------
    def draw_ar_obj(self, frame, rvec, tvec, marker_length=0.05):
        if self.camera_matrix is None or self.obj_vertices.size == 0:
            return frame

        verts = self.obj_vertices.copy()
        verts = verts - np.mean(verts, axis=0)
        max_dim = np.max(np.ptp(verts, axis=0))
        if max_dim == 0: return frame
        verts = verts / max_dim
        verts = verts * (marker_length * 2.0)

        Rx = np.array([[1, 0, 0],
                       [0, 0,-1],
                       [0, 1, 0]], dtype=np.float32)
        verts = verts.dot(Rx.T)

        verts[:, 1] -= np.min(verts[:, 1])

        imgpts, _ = cv2.projectPoints(
            verts, rvec, tvec, self.camera_matrix, self.dist
        )
        imgpts = np.int32(np.round(imgpts)).reshape(-1, 2)

        for face in self.obj_faces:
            pts = imgpts[face].reshape(-1, 1, 2)
            cv2.fillConvexPoly(frame, pts, (0, 200, 0))
            cv2.polylines(frame, [pts], True, (0, 100, 0), 1)

        return frame

    def run_ar_mode(self):
        if self.camera_matrix is None:
            print("No calibration found. Press K to calibrate first.")
            return
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        print("Running AR mode. Press ESC to exit.")
        while True:
            ret, frame = self.cam.read()
            if not ret: break
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05,
                                self.camera_matrix, self.dist)
                for i in range(len(ids)):
                    frame = self.draw_ar_obj(frame, rvecs[i], tvecs[i])
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    # ---------- TRANSFORM ----------
    def handle_transform_mode(self, frame):
        if not self.transform_mode or self.active_trackbar_mode != "TRANSFORM":
            return frame

        h, w = frame.shape[:2]
        dx = cv2.getTrackbarPos('Translate X', 'Camera') - 50
        dy = cv2.getTrackbarPos('Translate Y', 'Camera') - 50
        angle = cv2.getTrackbarPos('Rotation', 'Camera')
        scale = cv2.getTrackbarPos('Scale x10', 'Camera') / 10.0

        M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M_trans, (w, h))

        M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        frame = cv2.warpAffine(frame, M_rot, (w, h))
        return frame

    # ---------- KEY HANDLING ----------
    def handle_key(self, key):
        if key == ord('1'): self.mode = "COLOR"
        elif key == ord('2'): self.mode = "GRAY"
        elif key == ord('3'): self.mode = "HSV"
        elif key == ord('a'): self.toggle_mode("ADJUST")
        elif key == ord('g'): self.toggle_mode("GAUSSIAN")
        elif key == ord('b'): self.toggle_mode("BILATERAL")
        elif key == ord('h'): self.show_hist = not self.show_hist
        elif key == ord('c'): self.toggle_mode("CANNY")  # keep 'c' for Canny
        elif key == ord('d'): self.toggle_mode("HOUGH")
        elif key == ord('t'): self.toggle_mode("TRANSFORM")
        elif key == ord('0'):
            if self.active_trackbar_mode == "TRANSFORM":
                cv2.setTrackbarPos('Translate X', 'Camera', 50)
                cv2.setTrackbarPos('Translate Y', 'Camera', 50)
                cv2.setTrackbarPos('Rotation', 'Camera', 0)
                cv2.setTrackbarPos('Scale x10', 'Camera', 10)
                print("Transform reset to default")
        # ---- Panorama keys ----
        elif key == ord('p'):   # Toggle Panorama mode
            if self.panorama_mode:
                # Turn OFF panorama
                self.panorama_mode = False
                self.reset_panorama()
                print("Panorama OFF")
                self.panorama_status = "Panorama OFF"
            else:
                # Turn ON panorama
                self.panorama_mode = True
                print("Panorama ON: Z=Capture | O=Build | X=Reset")
                self.panorama_status = "Panorama ON"
            self.status_counter = 90

        elif key == ord('z') and self.panorama_mode:
            self.capture_panorama_frame()
        elif key == ord('o') and self.panorama_mode:
            self.build_panorama()
        elif key == ord('x') and self.panorama_mode:
            self.reset_panorama()



        # ---- Other features ----
        elif key == ord('k'): self.calibrate_camera()
        elif key == ord('r'): self.run_ar_mode()
        elif key == ord('q'): return False
        return True


    # ---------- MODES ----------
    def toggle_mode(self, mode):
        if self.active_trackbar_mode == mode:
            self.remove_trackbars()
            if mode == "ADJUST": self.show_adjust = False
            if mode == "GAUSSIAN": self.show_gaussian = False
            if mode == "BILATERAL": self.show_bilateral = False
            if mode == "TRANSFORM": self.transform_mode = False
        else:
            self.remove_trackbars()
            self.show_adjust = self.show_gaussian = self.show_bilateral = self.transform_mode = False

            if mode == "ADJUST":
                self.create_trackbars_adjust(); self.show_adjust = True
            elif mode == "GAUSSIAN":
                self.create_trackbars_gaussian(); self.show_gaussian = True
            elif mode == "BILATERAL":
                self.create_trackbars_bilateral(); self.show_bilateral = True
            elif mode == "CANNY":
                self.create_trackbars_canny(); self.show_canny = True
            elif mode == "HOUGH":
                self.create_trackbars_hough_lines(); self.hough_transform = True
            elif mode == "TRANSFORM":
                self.create_trackbars_transform(); self.transform_mode = True
            self.active_trackbar_mode = mode

    # ---------- PIPELINE ----------
    def handle_adjust_mode(self):
        if self.show_adjust and self.active_trackbar_mode == "ADJUST":
            self.alpha = cv2.getTrackbarPos('Alpha x0.1', 'Camera') / 10.0
            self.beta = cv2.getTrackbarPos('Beta', 'Camera') - 100

    def handle_gaussian_mode(self, frame):
        if self.show_gaussian and self.active_trackbar_mode == "GAUSSIAN":
            ksize = cv2.getTrackbarPos('Kernel Size', 'Camera')
            sigmaX = cv2.getTrackbarPos('SigmaX', 'Camera')
            if ksize % 2 == 0: ksize += 1
            if ksize < 1: ksize = 1
            return cv2.GaussianBlur(frame, (ksize, ksize), sigmaX)
        return frame

    def handle_bilateral_mode(self, frame):
        if self.show_bilateral and self.active_trackbar_mode == "BILATERAL":
            diameter = cv2.getTrackbarPos('Diameter', 'Camera')
            sigmaColor = cv2.getTrackbarPos('SigmaColor', 'Camera')
            sigmaSpace = cv2.getTrackbarPos('SigmaSpace', 'Camera')
            if diameter < 1: diameter = 1
            return cv2.bilateralFilter(frame, diameter, sigmaColor, sigmaSpace)
        return frame

    def handle_canny_mode(self, frame):
        if self.show_canny and self.active_trackbar_mode == "CANNY":
            t1 = cv2.getTrackbarPos('Threshold1', 'Camera')
            t2 = cv2.getTrackbarPos('Threshold2', 'Camera')
            return cv2.Canny(frame, t1, t2)
        return frame

    def handle_hough_lines(self, frame):
        if self.hough_transform and self.active_trackbar_mode == "HOUGH":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            threshold = cv2.getTrackbarPos('Threshold', 'Camera')
            min_line_length = cv2.getTrackbarPos('Min Line Length', 'Camera')
            max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Camera')
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                                    minLineLength=min_line_length, maxLineGap=max_line_gap)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def handle_histogram_mode(self, adj_frame):
        if self.show_hist:
            hist_img = self.show_histogram(adj_frame)
            cv2.imshow('Histogram', hist_img)
            if cv2.getWindowProperty('Histogram', cv2.WND_PROP_VISIBLE) < 1:
                self.show_hist = False
        else:
            if cv2.getWindowProperty('Histogram', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow('Histogram')

    def handle_running_mode(self, adj_frame):
        if self.mode == "GRAY": return self.process_gray(adj_frame)
        elif self.mode == "HSV": return self.process_hsv(adj_frame)
        return self.process_color(adj_frame)
    
        # ---------- PANORAMA ----------
    def capture_panorama_frame(self):
        ret, frame = self.cam.read()
        if ret:
            self.panorama_frames.append(frame.copy())
            msg = f"Captured frame {len(self.panorama_frames)}"
            print(msg)
            self.panorama_status = msg
            self.status_counter = 60  # show overlay ~2 sec

    def build_panorama(self):
        if len(self.panorama_frames) < 2:
            print("Need at least 2 frames to build panorama")
            self.panorama_status = "Need >=2 frames"
            self.status_counter = 60
            return

        base = self.panorama_frames[0]
        for i in range(1, len(self.panorama_frames)):
            nxt = self.panorama_frames[i]
            base = self.stitch_pair(base, nxt)

        # Save to output folder
        os.makedirs("output", exist_ok=True)
        filename = f"output/panorama_{cv2.getTickCount()}.jpg"
        cv2.imwrite(filename, base)
        print(f"Panorama built and saved: {filename}")
        self.panorama_status = f"Panorama saved: {filename}"
        self.status_counter = 120

        cv2.imshow("Panorama", base)

    def stitch_pair(self, img1, img2):
        gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(2000)
        k1, d1 = orb.detectAndCompute(gray1, None)
        k2, d2 = orb.detectAndCompute(gray2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(d1, d2), key=lambda x: x.distance)[:200]

        if len(matches) < 10:
            print(" Not enough matches to stitch")
            return img1

        src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Debug print sizes
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        print(f"Stitching img1: {w1}x{h1}, img2: {w2}x{h2}")

        # Safe canvas size (prevent overflow)
        max_width = 2000
        max_height = 1000
        panorama = cv2.warpPerspective(img1, H, (min(w1 + w2, max_width), min(max(h1, h2), max_height)))

        # Place second image
        panorama[0:h2, 0:w2] = img2

        # Crop black borders
        gray_pan = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_pan, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            panorama = panorama[y:y+h, x:x+w]

        return panorama



    def reset_panorama(self):
        self.panorama_frames = []
        print("Panorama frames cleared.")
        self.panorama_status = "Panorama reset"
        self.status_counter = 60

    def draw_status(self, frame):
        if self.status_counter > 0 and self.panorama_status:
            cv2.putText(frame, self.panorama_status, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            self.status_counter -= 1



    # ---------- MAIN LOOP ----------
    def run(self):
        running = True
        while running:
            ret, frame = self.cam.read()
            if not ret: break

            key = cv2.waitKey(1) & 0xFF
            running = self.handle_key(key)

            self.handle_adjust_mode()
            adj_frame = cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
            adj_frame = self.handle_gaussian_mode(adj_frame)
            adj_frame = self.handle_bilateral_mode(adj_frame)
            adj_frame = self.handle_canny_mode(adj_frame)
            adj_frame = self.handle_hough_lines(adj_frame)
            adj_frame = self.handle_transform_mode(adj_frame)
            display_frame = self.handle_running_mode(adj_frame)

            self.draw_help_text(display_frame)
            self.draw_status(display_frame)
            cv2.imshow('Camera', display_frame)
            self.handle_histogram_mode(adj_frame)

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = CameraApp()
    app.run()
