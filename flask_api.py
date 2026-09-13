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
    data = request.get_json()
    b64_image = data["base64"]
    template_name = data["template"]  # যেমন: "Weekly Test - Science.json"

    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_image)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        return jsonify({"success": False, "message": f"Template পাওয়া যায়নি: {template_name}"}), 400

    tmp_img_path = "/tmp/upload.jpg"
    cv2.imwrite(tmp_img_path, img)

    result = read_omr_sheet(tmp_img_path, template_path)
    return jsonify({"success": True, **result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
