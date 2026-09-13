"""
OpenCV OMR Reader — Gemini এর replacement
=============================================
Deterministic, pixel-based bubble detection। কোনো AI "অনুমান" নেই,
তাই একই ছবি বারবার দিলে সবসময় একই ফলাফল দেবে (Gemini এর মতো
non-deterministic না)। ১০০ ভাগ guaranteed accuracy কোনো পদ্ধতিতেই
নেই (কালির দাগ, halfway-marked বাবল, খুব বাঁকা ছবি এসব সবসময়ই
সমস্যা করতে পারে) — তাই confidence score সহ output দেওয়া হয়,
আর কম-confidence প্রশ্নগুলো "REVIEW" হিসেবে ফ্ল্যাগ হয় যাতে
admin এক নজরে শুধু সেগুলো চেক করে নিতে পারেন। বাস্তবে এই hybrid
approach-ই সবচেয়ে বেশি reliable — pure-automatic কোনো OMR
system-ই (দামি hardware scanner-ও) মানুষের চেক ছাড়া claim করে না
যে ১০০% নির্ভুল।

ব্যবহার:
    from omr_reader import read_omr_sheet
    result = read_omr_sheet("scanned_sheet.jpg", "template.json")
"""

import cv2
import json
import numpy as np


# ---------- ১. পাতা খুঁজে বের করে perspective সোজা করা ----------
def align_sheet(img, target_size=None):
    """
    শিটের বাইরের ৪ কোণা খুঁজে বের করে top-down (সোজা) ভিউতে
    warp করে। ফোনে বাঁকা করে তোলা ছবিতেও bubble গ্রিড সবসময়
    একই জায়গায় আসবে।
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sheet_contour = None
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.3 * img.shape[0] * img.shape[1]:
            sheet_contour = approx.reshape(4, 2)
            break

    if sheet_contour is None:
        # পাতার বর্ডার স্পষ্ট না পেলে, ছবিটাকে যেমন আছে তেমনই ব্যবহার করা হবে
        # (তখন ধরে নেওয়া হচ্ছে ছবি আগে থেকেই মোটামুটি সোজা/cropped)
        return img

    pts = order_points(sheet_contour)
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    return warped


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


# ---------- ২. Template অনুযায়ী প্রতিটা বাবলের অবস্থান বের করা ----------
def bubble_grid(block):
    x0, y0 = block["first_bubble"]
    x1, y1 = block["last_bubble"]
    n_q, n_opt = block["questions"], block["options"]

    grid = []
    for q in range(n_q):
        row = []
        for o in range(n_opt):
            # প্রথম আর শেষ বাবলের মাঝে সমান ভাগে বাকিগুলোর অবস্থান বের করা
            x = x0 + (x1 - x0) * (o / (n_opt - 1)) if n_opt > 1 else x0
            y = y0 + (y1 - y0) * (q / (n_q - 1)) if n_q > 1 else y0
            row.append((x, y))
        grid.append(row)
    return grid


# ---------- ৩. প্রতিটা বাবলের darkness মেপে filled/blank ঠিক করা ----------
def read_block(gray, block, sample_radius=9, fill_gap_threshold=20):
    grid = bubble_grid(block)
    answers = []
    for q_idx, row in enumerate(grid):
        darks = []
        for (x, y) in row:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), sample_radius, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]   # কম মান = গাঢ় = সম্ভবত ভরাট
            darks.append(mean_val)

        darks = np.array(darks)
        sorted_idx = np.argsort(darks)
        darkest, second_darkest = darks[sorted_idx[0]], darks[sorted_idx[1]]
        gap = second_darkest - darkest

        options = ["A", "B", "C", "D", "E"][: block["options"]]

        if gap < fill_gap_threshold * 0.4:
            # কোনো বাবলই বাকিগুলোর চেয়ে স্পষ্টভাবে গাঢ় না -> সম্ভবত ফাঁকা
            result, confidence = "BLANK", "high"
        elif gap < fill_gap_threshold:
            # একটা বাবল একটু গাঢ়, কিন্তু পার্থক্য কম -> হালকা দাগ/সন্দেহজনক
            result, confidence = options[sorted_idx[0]], "low"
        else:
            result, confidence = options[sorted_idx[0]], "high"

        # multiple-mark check: প্রথম আর দ্বিতীয় গাঢ় বাবল দুটোই যথেষ্ট গাঢ় হলে duplicate mark
        if gap < fill_gap_threshold and second_darkest < darks.mean() - fill_gap_threshold:
            result, confidence = "MULTIPLE", "low"

        answers.append({
            "qNo": q_idx + 1,
            "answer": result,
            "confidence": confidence,
            "raw": [round(d, 1) for d in darks],
        })
    return answers


# ---------- ৪. Roll Number গ্রিড পড়া (column-wise: digit position x value 0-9) ----------
def read_roll(gray, roll_block, sample_radius=9, fill_gap_threshold=20):
    x0, y0 = roll_block["first_bubble"]   # কলাম-১, মান 0 (উপরের সারি)
    x1, y1 = roll_block["last_bubble"]    # শেষ কলাম, মান 9 (নিচের সারি)
    n_digits = roll_block["digits"]
    n_values = 10  # 0-9

    digits = []
    low_confidence_cols = []
    for col in range(n_digits):
        x = x0 + (x1 - x0) * (col / (n_digits - 1)) if n_digits > 1 else x0
        darks = []
        for val in range(n_values):
            y = y0 + (y1 - y0) * (val / (n_values - 1))
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), sample_radius, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            darks.append(mean_val)
        darks = np.array(darks)
        order = np.argsort(darks)
        gap = darks[order[1]] - darks[order[0]]
        digit = str(order[0]) if gap > fill_gap_threshold else "?"
        if gap <= fill_gap_threshold:
            low_confidence_cols.append(col + 1)
        digits.append(digit)

    return "".join(digits), low_confidence_cols


# ---------- ৫. মূল ফাংশন ----------
def read_omr_sheet(image_path, template_path):
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)

    img = cv2.imread(image_path)
    img = align_sheet(img)

    # টেমপ্লেট যে সাইজে ক্যালিব্রেট হয়েছিল, সেই সাইজে resize করা —
    # তাহলে ক্যালিব্রেশনের coordinate গুলো নতুন ছবির সাথেও মিলবে
    tw, th = template["image_size"]
    img = cv2.resize(img, (tw, th))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    result = {"subjects": [], "needs_review": [], "roll": "", "roll_confidence": "high"}

    if "roll" in template:
        roll_str, low_cols = read_roll(gray, template["roll"])
        result["roll"] = roll_str
        if low_cols or "?" in roll_str:
            result["roll_confidence"] = "low"

    for block in template["blocks"]:
        answers = read_block(gray, block)
        result["subjects"].append({"subject": block["subject"], "questions": answers})
        for a in answers:
            if a["confidence"] == "low":
                result["needs_review"].append(
                    {"subject": block["subject"], "qNo": a["qNo"], "answer": a["answer"]}
                )
    return result


if __name__ == "__main__":
    import sys
    out = read_omr_sheet(sys.argv[1], sys.argv[2])
    print(json.dumps(out, indent=2, ensure_ascii=False))
