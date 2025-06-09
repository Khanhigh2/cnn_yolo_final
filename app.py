print("=== Defect Detection Backend ===")
print("Starting server...")

import os
import sys

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    print("✅ Flask imports successful")
except Exception as e:
    print(f"❌ Flask import error: {e}")
    exit(1)

try:
    import cv2
    import numpy as np
    import base64
    from datetime import datetime
    print("✅ OpenCV and other imports successful")
except Exception as e:
    print(f"❌ OpenCV import error: {e}")
    exit(1)

try:
    from ultralytics import YOLO
    print("✅ YOLO import successful")
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"❌ YOLO import error: {e}")
    YOLO_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Load YOLO model
MODEL_PATH = r"c:\project_root\runs\detect\yolo_inventory\weights\best.pt"
print(f"🔄 Loading model from: {MODEL_PATH}")

model = None
if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        model = None
else:
    print("⚠️ YOLO model not loaded")

# Simple in-memory storage
history = []

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        # Use absolute path to ensure file is found
        html_path = os.path.join(os.path.dirname(__file__), 'frontend.html')
        print(f"🔍 Looking for HTML file at: {html_path}")
        
        if not os.path.exists(html_path):
            print(f"❌ HTML file not found at: {html_path}")
            return "HTML file not found", 404
            
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"✅ Successfully loaded HTML file ({len(content)} chars)")
            return content
    except Exception as e:
        print(f"❌ Error loading HTML file: {e}")
        return f"Error loading HTML file: {e}", 500

@app.route('/test')
def test_page():
    """Test page to verify server works"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <style>
            body { font-family: Arial; padding: 50px; background: #f0f0f0; }
            .card { background: white; padding: 30px; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🎉 Server Working!</h1>
            <p>If you see this page, the Flask server is working correctly.</p>
            <p><a href="/">Go to Main App</a></p>
        </div>
    </body>
    </html>
    """

def base64_to_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def image_to_base64(image):
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

@app.route('/api/test', methods=['GET'])
def test():
    print("🧪 Test endpoint called")
    return jsonify({
        "status": "OK",
        "message": "Backend server is running!",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect():
    print("🔍 Detect endpoint called")
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data"}), 400
        
        print("📸 Image received for processing")
        
        # Decode image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        if model is None:
            # Fallback response when YOLO is not available
            result = {
                "success": True,
                "has_defects": False,
                "message": "Không phát hiện ra lỗi (YOLO model chưa load)"
            }
        else:
            # Use YOLO model for detection
            try:
                print("🤖 Running YOLO inference...")
                results = model(image)
                  # Check if any defects detected
                detections = results[0].boxes
                has_defects = len(detections) > 0 if detections is not None else False
                
                if has_defects:
                    print(f"🔍 Found {len(detections)} defects")
                    
                    # Draw bounding boxes and collect defect info
                    annotated_image = image.copy()
                    defect_types = []
                    
                    for box in detections:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        
                        # Get defect name from model
                        defect_name = model.names[cls_id] if cls_id < len(model.names) else f"Unknown_{cls_id}"
                        defect_types.append(f"{defect_name}({confidence:.2f})")
                        
                        # Draw rectangle with thicker line
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Add label with background
                        label = f"{defect_name} {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Draw label background
                        cv2.rectangle(annotated_image, (x1, y1-30), (x1+label_size[0]+10, y1), (0, 0, 255), -1)
                        # Draw label text
                        cv2.putText(annotated_image, label, (x1+5, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert annotated image to base64
                    annotated_base64 = image_to_base64(annotated_image)
                    
                    result = {
                        "success": True,
                        "has_defects": True,
                        "defect_count": len(detections),
                        "defect_types": defect_types,
                        "message": f"Phát hiện {len(detections)} lỗi trên sản phẩm",
                        "annotated_image": annotated_base64
                    }
                    
                    # Add to history
                    history.append({
                        "timestamp": datetime.now().isoformat(),
                        "defect_count": len(detections),
                        "defect_types": defect_types,
                        "confidence": float(detections[0].conf[0]) if len(detections) > 0 else 0,
                        "image": annotated_base64
                    })
                    
                else:
                    result = {
                        "success": True,
                        "has_defects": False,
                        "message": "Không phát hiện ra lỗi"
                    }
                    print("✅ No defects detected")
                    
            except Exception as e:
                print(f"❌ YOLO inference error: {e}")
                result = {
                    "success": False,
                    "error": f"Model inference failed: {str(e)}"
                }
        
        print(f"✅ Detection result: {result.get('message', 'Processing complete')}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    print("📜 History endpoint called")
    return jsonify({
        "success": True,
        "history": history
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Defect Detection API",
        "status": "running",
        "endpoints": [
            "GET /api/test",
            "POST /api/detect", 
            "GET /api/history"
        ]
    })

if __name__ == '__main__':
    print("🚀 Starting Flask development server...")
    print("🌐 Server URL: http://localhost:5000")
    print("📋 Available endpoints:")
    print("   GET  /api/test    - Test server status")
    print("   POST /api/detect  - Detect defects in image")
    print("   GET  /api/history - Get detection history")
    print("🔄 Press Ctrl+C to stop the server")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
