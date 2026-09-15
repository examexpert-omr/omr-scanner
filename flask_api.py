"""
Flask API — আপনার Apps Script থেকে কল করার জন্য
====================================================
এটা Render.com / Railway.app এর ফ্রি tier এ deploy করা যায়।

deploy করার পর, Apps Script এর detectMCQOMR() ফাংশনে
Gemini API call-এর জায়গায় এই URL কল করবেন (নিচে integration
নোট দেওয়া আছে)।

লোকাল টেস্ট: python 3_flask_api.py
"""

from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import json
import os

from omr_reader import read_omr_sheet

app = Flask(__name__)

TEMPLATES_DIR = "templates"  # প্রতিটা category+group এর জন্য আলাদা template.json এখানে রাখবেন


@app.route("/scan-omr", methods=["POST"])
def scan_omr():
    try:
        data = request.get_json()
        if not data or "base64" not in data or "template" not in data:
            return jsonify({"success": False, "message": "Invalid request payload"}), 400

        b64_image = data["base64"]
        template_name = data["template"]  # যেমন: "Arts.json"[cite: 4, 5]

        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1][cite: 5]
        
        img_bytes = base64.b64decode(b64_image)[cite: 5]
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)[cite: 5]
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)[cite: 5]

        if img is None:
            return jsonify({"success": False, "message": "ছবি ডিকোড করা সম্ভব হয়নি।"}), 400

        template_path = os.path.join(TEMPLATES_DIR, template_name)[cite: 5]
        if not os.path.exists(template_path):
            return jsonify({"success": False, "message": f"Template পাওয়া যায়নি: {template_name}"}), 400[cite: 5]

        tmp_img_path = "/tmp/upload.jpg"[cite: 5]
        cv2.imwrite(tmp_img_path, img)[cite: 5]

        # OMR Processing
        result = read_omr_sheet(tmp_img_path, template_path)[cite: 5]
        return jsonify({"success": True, **result})[cite: 5]

    except Exception as e:
        # কোনো কোড এরর হলে ৫০০ এর বদলে সঠিক মেসেজ ব্যাক করবে
        return jsonify({"success": False, "message": f"Server Error: {str(e)}"}), 200
