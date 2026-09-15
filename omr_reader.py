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
import numpy as np

def align_sheet(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding দিয়ে কালো চারকোনা এলাইনমেন্ট মার্কগুলো খুঁজে বের করা
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marker_centers = []
    img_area = img.shape[0] * img.shape[1]

    for c in contours:
        area = cv2.contourArea(c)
        # চারকোনা বর্ডার বক্সের সাইজ ফিল্টারিং (খুব ছোট বা বড় কন্টুর বাদ দেয়া)
        if 0.0001 * img_area < area < 0.01 * img_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4:  # চারকোনা আকৃতি
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    marker_centers.append([cx, cy])

    # যদি ঠিক ৪টি অথবা তার বেশি এলাইনমেন্ট মার্ক পায়
    if len(marker_centers) >= 4:
        pts = np.array(marker_centers, dtype="float32")
        rect = order_points(pts[:4])  # top-left, top-right, bottom-right, bottom-left
        
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxW - 1, 0],
            [maxW - 1, maxH - 1],
            [0, maxH - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxW, maxH))
        return warped

    # এলাইনমেন্ট মার্ক না পেলে আগের নিয়মে ব্যাকআপ এলাইনমেন্ট কাজ করবে[cite: 6]
    return fallback_align(img)

def fallback_align(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.3 * img.shape[0] * img.shape[1]:
            pts = order_points(approx.reshape(4, 2))[cite: 6]
            (tl, tr, br, bl) = pts
            maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(img, M, (maxW, maxH))

    return img


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
