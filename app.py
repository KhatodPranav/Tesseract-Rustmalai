from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import io
import os
import base64
import time

app = Flask(__name__)

# Load your model
MODEL_PATH = r"C:\Users\Pranav\Desktop\sixray_v3\runs\detect\cargosight_yolo117\weights\best.pt"
model = YOLO(MODEL_PATH)

dashboard_stats = {
    "total_scans": 0,
    "clean_scans": 0, 
    "anomalies_found": 0,
    "re_scans": 0
}

# Define how dangerous each item actually is
THREAT_WEIGHTS = {
    'Gun': 1.0,      # Critical Threat: 100% of confidence becomes Risk
    'Knife': 0.85,   # High Threat: 85% of confidence becomes Risk
    'Scissors': 0.50,# Medium Threat: 50% of confidence becomes Risk
    'Pliers': 0.25,  # Low Threat (Tools): 25% of confidence becomes Risk
    'Wrench': 0.25   # Low Threat (Tools): 25% of confidence becomes Risk
}

# Create feedback folders automatically
FEEDBACK_FOLDERS = ["object_in_object", "undetectable", "half_object_detected"]
for folder in FEEDBACK_FOLDERS:
    os.makedirs(os.path.join("feedback_data", folder), exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    results = model.predict(source=img)
    
    detections = []
    total_risk_score = 0.0  # Start with 0 risk
    
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            conf_percent = round(float(box.conf[0]) * 100, 2)
            
            # --- THE NEW RISK MATH ---
            # Get the weight (default to 0.5 if it's a new unknown class)
            weight = THREAT_WEIGHTS.get(class_name, 0.5) 
            # Multiply confidence by the danger weight and add it to the total bag risk
            total_risk_score += (conf_percent * weight)
            
            detections.append({
                "class": class_name,
                "confidence": conf_percent
            })

    # Cap the maximum risk at 100%
    final_risk = min(100.0, round(total_risk_score, 1))

    # Update Dashboard Stats
    dashboard_stats["total_scans"] += 1
    if len(detections) > 0:
        dashboard_stats["anomalies_found"] += 1
    else:
        dashboard_stats["clean_scans"] += 1 

    # Draw boxes and convert to Base64
    annotated_img_array = results[0].plot() 
    annotated_pil = Image.fromarray(annotated_img_array[..., ::-1])
    
    buffered = io.BytesIO()
    annotated_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        "image": "data:image/jpeg;base64," + img_base64,
        "detections": detections,
        "stats": dashboard_stats,
        "risk_score": final_risk
    })

@app.route('/feedback', methods=['POST'])
def save_feedback():
    if 'image' not in request.files or 'reason' not in request.form:
        return jsonify({"error": "Missing data"}), 400

    file = request.files['image']
    reason = request.form['reason']
    
    filename = f"re_scan_{int(time.time())}.jpg"
    save_path = os.path.join("feedback_data", reason, filename)
    file.save(save_path)
    
    dashboard_stats["re_scans"] += 1
    return jsonify({"message": "Saved to " + reason, "stats": dashboard_stats})

if __name__ == '__main__':
    app.run(debug=True, port=5000)