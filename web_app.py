"""Web-based real-time hand gesture recognition and occlusion stress-test dashboard."""

import time
import os
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

from src.feature_extractor import HandLandmarkExtractor
from src.occluder import apply_block_occlusion
from src.model import load_model_bundle, DEFAULT_MODEL_PATH

app = Flask(__name__)

# Global state for web controls
STATE = {
    "occlusion_pct": 0.0,
    "show_skeleton": True,
    "predicted_gesture": "INITIALIZING...",
    "confidence": 0.0,
    "tracking_status": "WAITING",
    "fps": 0.0,
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hand Gesture Recognition Live Test</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px;
    }
    header {
      text-align: center;
      margin-bottom: 20px;
    }
    h1 {
      font-size: 24px;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 6px;
    }
    p.subtitle {
      color: #94a3b8;
      font-size: 14px;
    }
    .main-container {
      display: flex;
      gap: 24px;
      max-width: 1200px;
      width: 100%;
      flex-wrap: wrap;
      justify-content: center;
    }
    .video-card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      position: relative;
    }
    .video-feed {
      border-radius: 8px;
      display: block;
      width: 640px;
      height: 480px;
      background: #000;
      object-fit: cover;
    }
    .controls-card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 24px;
      width: 360px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .section-title {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #94a3b8;
      margin-bottom: 8px;
    }
    .badge {
      display: inline-block;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 18px;
      font-weight: 700;
      background: #0284c7;
      color: #fff;
    }
    .btn-group {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }
    button {
      background: #334155;
      border: 1px solid #475569;
      color: #f8fafc;
      padding: 10px 8px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    button:hover {
      background: #0ea5e9;
      border-color: #38bdf8;
    }
    button.active {
      background: #0284c7;
      border-color: #38bdf8;
      color: #fff;
    }
    .slider-container {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    input[type=range] {
      width: 100%;
      accent-color: #0ea5e9;
      cursor: pointer;
    }
    .info-box {
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 12px;
      font-size: 13px;
      line-height: 1.5;
      color: #cbd5e1;
    }
    .tag-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .tag {
      background: #1e293b;
      border: 1px solid #475569;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      color: #38bdf8;
    }
  </style>
</head>
<body>
  <header>
    <h1>Hand Gesture Recognition & Occlusion Stress-Tester</h1>
    <p class="subtitle">Live webcam feed with real-time MediaPipe 21-landmark tracking and synthetic occlusion injection</p>
  </header>

  <div class="main-container">
    <div class="video-card">
      <img class="video-feed" src="/video_feed" alt="Live Camera Feed">
    </div>

    <div class="controls-card">
      <div>
        <div class="section-title">Tested Gestures</div>
        <div class="tag-list">
          <span class="tag">Thumbs Up</span>
          <span class="tag">Thumbs Down</span>
          <span class="tag">Peace</span>
          <span class="tag">OK Sign</span>
          <span class="tag">Pointing</span>
          <span class="tag">Fist</span>
        </div>
      </div>

      <div>
        <div class="section-title">Synthetic Occlusion Level</div>
        <div class="btn-group">
          <button onclick="setOcclusion(0.0)" class="active" id="btn-0">0%</button>
          <button onclick="setOcclusion(0.10)" id="btn-10">10%</button>
          <button onclick="setOcclusion(0.20)" id="btn-20">20%</button>
          <button onclick="setOcclusion(0.30)" id="btn-30">30%</button>
          <button onclick="setOcclusion(0.45)" id="btn-45">45%</button>
          <button onclick="setOcclusion(0.60)" id="btn-60">60%</button>
        </div>
      </div>

      <div class="slider-container">
        <label for="occSlider" style="font-size:13px; color:#94a3b8; display:flex; justify-content:space-between;">
          <span>Custom Occlusion:</span>
          <span id="occVal" style="color:#38bdf8; font-weight:700;">0%</span>
        </label>
        <input type="range" id="occSlider" min="0" max="60" value="0" step="5" oninput="sliderChange(this.value)">
      </div>

      <div>
        <button style="width: 100%;" onclick="toggleSkeleton()">Toggle Skeleton Overlay</button>
      </div>

      <div class="info-box">
        <strong>Benchmarking Note:</strong>
        <p style="margin-top: 4px;">Notice how at 0%-20% occlusion, detection remains robust. At 45%-60%, key landmark topology is lost, triggering failure modes.</p>
      </div>
    </div>
  </div>

  <script>
    function setOcclusion(val) {
      fetch('/set_occlusion?val=' + val);
      document.getElementById('occSlider').value = val * 100;
      document.getElementById('occVal').innerText = Math.round(val * 100) + '%';
      document.querySelectorAll('.btn-group button').forEach(b => b.classList.remove('active'));
      let b = document.getElementById('btn-' + Math.round(val * 100));
      if (b) b.classList.add('active');
    }
    function sliderChange(pct) {
      let val = pct / 100.0;
      fetch('/set_occlusion?val=' + val);
      document.getElementById('occVal').innerText = pct + '%';
      document.querySelectorAll('.btn-group button').forEach(b => b.classList.remove('active'));
    }
    function toggleSkeleton() {
      fetch('/toggle_skeleton');
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/set_occlusion")
def set_occlusion():
    val = float(request.args.get("val", 0.0))
    STATE["occlusion_pct"] = np.clip(val, 0.0, 0.8)
    return jsonify({"status": "ok", "occlusion_pct": STATE["occlusion_pct"]})


@app.route("/toggle_skeleton")
def toggle_skeleton():
    STATE["show_skeleton"] = not STATE["show_skeleton"]
    return jsonify({"status": "ok", "show_skeleton": STATE["show_skeleton"]})


def generate_frames():
    bundle = load_model_bundle(DEFAULT_MODEL_PATH)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    cap = cv2.VideoCapture(0)
    extractor = HandLandmarkExtractor(static_image_mode=False)

    last_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = 1.0 / dt
            last_time = now

            occ_pct = STATE["occlusion_pct"]

            # Hand bounding box
            hand_bbox = extractor.extract_hand_bbox(frame)

            if occ_pct > 0.0:
                display_frame = apply_block_occlusion(
                    image=frame,
                    occlusion_pct=occ_pct,
                    occlude_roi="random_hand_patch",
                    hand_bbox=hand_bbox,
                    occlusion_type="black",
                )
            else:
                display_frame = frame.copy()

            landmarks = extractor.extract_landmarks(display_frame, normalize=True)

            pred_text = "NO HAND"
            conf = 0.0
            status_text = "SEARCHING"
            status_col = (0, 0, 255)

            if landmarks is not None and len(landmarks) == 63:
                status_text = "LOCKED"
                status_col = (0, 255, 0)
                features = landmarks.reshape(1, -1)
                code = model.predict(features)[0]
                pred_text = label_encoder.inverse_transform([code])[0].upper()
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                    conf = float(np.max(probs)) * 100.0
                else:
                    conf = 100.0

                if STATE["show_skeleton"]:
                    display_frame, _ = extractor.draw_landmarks_on_image(display_frame)
            else:
                if occ_pct > 0.0 and hand_bbox is not None:
                    status_text = "OCCLUDED / LOST"
                    status_col = (0, 140, 255)

            # Overlay Header
            cv2.rectangle(display_frame, (0, 0), (w, 85), (15, 23, 42), -1)
            cv2.line(display_frame, (0, 85), (w, 85), (56, 189, 248), 2)

            cv2.putText(
                display_frame,
                f"GESTURE: {pred_text}",
                (15, 36),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (56, 189, 248) if status_text == "LOCKED" else (148, 163, 184),
                2,
                cv2.LINE_AA,
            )

            if conf > 0:
                cv2.putText(
                    display_frame,
                    f"Conf: {conf:.1f}%",
                    (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (74, 222, 128),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display_frame,
                f"Status: {status_text} | Occlusion: {int(occ_pct * 100)}% | FPS: {fps:.1f}",
                (w - 380, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_col,
                2,
                cv2.LINE_AA,
            )

            ret2, buffer = cv2.imencode(".jpg", display_frame)
            if not ret2:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        cap.release()
        extractor.close()


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("\n========================================================")
    print(" Starting Web-based Hand Gesture Recognition Live Demo")
    print(" Open in your browser: http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
