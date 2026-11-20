"""
AquaSense AI — Smart Aquaculture Monitoring & Prediction System
================================================================
Unified Application Entry Point & Web Server:
  1. Real-Time Raspberry Pi Sensor Acquisition (DS18B20 & ADS1115)
  2. Benson-LSTM / Benson Physical DLEVS Model for Calculating DO
  3. QC-LSTM 10-Step Sequence Analysis for Water Quality (GOOD / BAD)
  4. ViT-B/16 Vision Transformer for Fish Disease Detection with Real Live Webcam
  5. Web GUI matching the AquaSense Dashboard
"""

from __future__ import annotations
import os
import sys
import time
import socket
import threading
import tempfile
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PIL import Image
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file

from services.data_provider import RaspberryPiSensorProvider
from services.model_service import ModelService

# ---------------------------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

print("\n[INIT] Initializing AquaSense Services...")
sensor_provider = RaspberryPiSensorProvider()
model_service = ModelService()

# ---------------------------------------------------------------------------
# Camera State
# ---------------------------------------------------------------------------
latest_frame = None
frame_lock = threading.Lock()
camera = None
camera_available = False

def init_camera():
    """
    Detects a real USB or CSI camera on Raspberry Pi.
    First tries /dev/video0-9 (most likely for real cameras).
    Falls back to all /dev/video* if none found.
    """
    global camera, camera_available
    if camera is not None and camera.isOpened():
        return

    is_linux = sys.platform.startswith("linux")

    if is_linux:
        import glob
        # Priority 1: /dev/video0-9 (typical USB webcam / CSI camera)
        primary   = sorted(glob.glob("/dev/video[0-9]"))
        # Priority 2: all other /dev/video* (some USB cams appear at higher indices)
        secondary = [d for d in sorted(glob.glob("/dev/video*")) if d not in primary]
        all_devices = primary + secondary

        candidates = []
        for path in all_devices:
            candidates.append((path, cv2.CAP_V4L2))
            candidates.append((path, cv2.CAP_ANY))
    else:
        candidates = [(i, cv2.CAP_ANY) for i in [0, 1, 2]]

    for dev, backend in candidates:
        try:
            cam = cv2.VideoCapture(dev, backend)
            if not cam.isOpened():
                cam.release()
                continue

            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Warm-up reads
            for _ in range(3):
                cam.read()
                time.sleep(0.05)

            ok, test_frame = cam.read()
            if ok and test_frame is not None and test_frame.size > 0:
                camera = cam
                camera_available = True
                backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "DEFAULT"
                print(f"[CAMERA] Connected → {dev}  backend={backend_name}")
                return
            cam.release()
        except Exception as e:
            continue   # silently skip — too many devices to log each

    camera = None
    camera_available = False
    print("[CAMERA] No real camera found. Manual image upload is available.")

init_camera()

def camera_stream_loop():
    """Continuously grabs live frames. Auto-reconnects if camera disconnects."""
    global latest_frame, camera, camera_available
    fail_count = 0
    while True:
        if camera is not None and camera.isOpened():
            ok, frame = camera.read()
            if ok and frame is not None and frame.size > 0:
                with frame_lock:
                    latest_frame = frame.copy()
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > 10:
                    print("[CAMERA] Stream lost. Reconnecting...")
                    try:
                        camera.release()
                    except Exception:
                        pass
                    camera = None
                    camera_available = False
                    fail_count = 0
                time.sleep(0.05)
        else:
            time.sleep(1.5)
            init_camera()
        time.sleep(0.03)

camera_thread = threading.Thread(target=camera_stream_loop, daemon=True)
camera_thread.start()


def generate_mjpeg_stream():
    """Streams live webcam frames to the browser. Never streams dummy test images."""
    while True:
        frame_bytes = None
        with frame_lock:
            if latest_frame is not None:
                ok, buffer = cv2.imencode(".jpg", latest_frame)
                if ok:
                    frame_bytes = buffer.tobytes()

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.03)
        else:
            # When camera is not ready, generate a clean live camera placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "Connecting to Live Camera...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )
            _, buffer = cv2.imencode(".jpg", placeholder)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.5)

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ---------------------------------------------------------------------------
# Flask Web Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sensor_data")
def sensor_data():
    reading = sensor_provider.get_live_reading()
    wq_prediction = model_service.predict_water_quality(reading)
    return jsonify({**reading, **wq_prediction})

@app.route("/water_reset", methods=["POST"])
def water_reset():
    model_service.reset_water_sequence()
    return jsonify({"success": True, "message": "Sequence reset to 0/10."})

@app.route("/camera_status")
def camera_status():
    """Returns whether a real camera is connected and streaming."""
    return jsonify({
        "available": camera_available,
        "streaming": latest_frame is not None
    })

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/captured_fish_image")
def captured_fish_image():
    path = os.path.join(BASE_DIR, "captured_fish.jpg")
    if os.path.isfile(path):
        return send_file(path, mimetype="image/jpeg")
    return Response(b"", mimetype="image/jpeg")

@app.route("/capture", methods=["POST"])
def capture_and_predict():
    """
    Captures the current live webcam frame and diagnoses it with ViT-B/16.
    Waits up to 3 seconds for a frame to be available before giving up.
    """
    frame = None

    # Wait up to 3 seconds for a live frame (handles slight thread startup delay)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                break
        time.sleep(0.1)

    if frame is None:
        return jsonify({
            "success": False,
            "message": "No live camera frame available. Please plug in a USB webcam and restart."
        }), 400

    try:
        # Save captured frame as jpeg
        captured_path = os.path.join(BASE_DIR, "captured_fish.jpg")
        cv2.imwrite(captured_path, frame)

        # Convert BGR → RGB → PIL for ViT model
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        result = model_service.predict_fish(pil_img)
        result["image_url"] = "/captured_fish_image"
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/predict_fish_image", methods=["POST"])
def predict_uploaded_image():
    """Receives manual fish image upload and runs the ViT model."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty file name."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        
        captured_path = os.path.join(BASE_DIR, "captured_fish.jpg")
        image.save(captured_path)

        result = model_service.predict_fish(image)
        result["image_url"] = "/captured_fish_image"
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------------------------------------------------------------------
# Server Startup & PuTTY Banner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000

    print("\n" + "=" * 76)
    print("                 AQUASENSE AI — SMART AQUACULTURE SYSTEM            ")
    print("=" * 76)
    print(" [✓] Real-Time Sensors:    Temperature, Turbidity, pH, Salinity (RPi ADS1115)")
    print(" [✓] Benson Model:         Active (Virtual Dissolved Oxygen Calculation)")
    print(" [✓] QC-LSTM Model:        Active (10-Step Water Quality Prediction)")
    print(" [✓] Live Webcam Stream:   Active (Streams directly to GUI)")
    print(" [✓] ViT-B/16 Model:       Active (Fish Disease Health Diagnosis)")
    print("-" * 76)
    print(" >>> OPEN THIS LINK IN CHROME TO VIEW THE GUI:")
    print(f"     http://{local_ip}:{port}")
    print(f"     http://localhost:{port}")
    print("=" * 76 + "\n")

    try:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    finally:
        if camera is not None and camera.isOpened():
            camera.release()
