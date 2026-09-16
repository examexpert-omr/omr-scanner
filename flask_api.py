"""
Flask API — আপনার Apps Script থেকে কল করার জন্য
====================================================
এটা Render.com এ deploy করা হয়েছে।

লোকাল টেস্ট: python flask_api.py
"""

from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import os

from omr_reader import read_omr_sheet

app = Flask(__name__)

TEMPLATES_DIR = "templates"  # প্রতিটা group এর জন্য আলাদা template.json এখানে রাখা হয়


@app.route("/")
def home():
    return "OMR Scanner is running!", 200


@app.route("/scan-omr", methods=["POST"])
def scan_omr():
    try:
        data = request.get_json()
        if not data or "base64" not in data or "template" not in data:
            return jsonify({"success": False, "message": "Invalid request payload"}), 400

        b64_image = data["base64"]
        template_name = data["template"]  # যেমন: "Arts.json"

        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_image)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "message": "ছবি ডিকোড করা সম্ভব হয়নি।"}), 400

        template_path = os.path.join(TEMPLATES_DIR, template_name)
        if not os.path.exists(template_path):
            return jsonify({"success": False, "message": f"Template পাওয়া যায়নি: {template_name}"}), 400

        tmp_img_path = "/tmp/upload.jpg"
        cv2.imwrite(tmp_img_path, img)

        result = read_omr_sheet(tmp_img_path, template_path)
        return jsonify({"success": True, **result})

    except Exception as e:
        # কোনো কোড এরর হলে ব্যাখ্যাসহ মেসেজ ফেরত পাঠানো হচ্ছে
        return jsonify({"success": False, "message": f"Server Error: {str(e)}"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
