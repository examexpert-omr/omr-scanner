"""
OpenCV OMR Reader (মার্কার-ভিত্তিক Alignment)
=================================================
আগের ভার্সনে পুরো ছবির বর্ডার/contour খুঁজে align করার চেষ্টা হতো,
যেটা background আলাদা হলে (বিশেষ করে ফোনের ছবিতে, যেখানে টেবিল/
পেছনের রং কাগজের কাছাকাছি উজ্জ্বল) ব্যর্থ হতো।

এই ভার্সনে বরং কাগজের নিজের ছাপানো ৪ কোণার কালো বর্গ (marker)
খুঁজে বের করে তার ভিত্তিতে perspective warp করা হয় — এটা
background/আলো/phone যাই হোক না কেন নির্ভরযোগ্যভাবে কাজ করে।

ব্যবহার:
    from omr_reader import read_omr_sheet
    result = read_omr_sheet("scanned_sheet.jpg", "template.json")
"""

import cv2
import json
import numpy as np

CANON_SIZE = (1200, 800)  # সব ছবি এই নির্দিষ্ট সাইজে warp হবে (calibration ও scan — দুটোতেই)


# ---------- ১. ৪ কোণার কালো বর্গ (marker) খুঁজে বের করা ----------
def detect_corner_markers(gray):
    h, w = gray.shape
    # Adaptive threshold — ছবির বিভিন্ন জায়গায় আলো কম-বেশি হলেও কাজ করে
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150 or area > 3000:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh == 0:
            continue
        ar = bw / float(bh)
        if 0.6 < ar < 1.6:
            solidity = area / float(bw * bh)
            if solidity > 0.55:
                candidates.append((x + bw / 2, y + bh / 2))

    if len(candidates) < 4:
        return None  # যথেষ্ট marker পাওয়া যায়নি

    corners_target = {"TL": (0, 0), "TR": (w, 0), "BL": (0, h), "BR": (w, h)}
    found = {}
    used = set()
    for name, (tx, ty) in corners_target.items():
        remaining = [p for p in candidates if p not in used]
        best = min(remaining, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)
        used.add(best)
        found[name] = best
    return found


# ---------- ২. মার্কার দিয়ে ছবি সোজা করে নির্দিষ্ট (canonical) সাইজে আনা ----------
def warp_by_markers(img, canon_size=CANON_SIZE):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    markers = detect_corner_markers(gray)

    if markers is None:
        # মার্কার খুঁজে পাওয়া না গেলে -> পুরো ছবিটাই canon সাইজে resize (fallback)
        return cv2.resize(img, canon_size), False

    cw, ch = canon_size
    src = np.array([markers["TL"], markers["TR"], markers["BR"], markers["BL"]], dtype="float32")
    dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (cw, ch))
    return warped, True


# ---------- ৩. Template অনুযায়ী প্রতিটা বাবলের অবস্থান বের করা ----------
def bubble_grid(block):
    x0, y0 = block["first_bubble"]
    x1, y1 = block["last_bubble"]
    n_q, n_opt = block["questions"], block["options"]

    grid = []
    for q in range(n_q):
        row = []
        for o in range(n_opt):
            x = x0 + (x1 - x0) * (o / (n_opt - 1)) if n_opt > 1 else x0
            y = y0 + (y1 - y0) * (q / (n_q - 1)) if n_q > 1 else y0
            row.append((x, y))
        grid.append(row)
    return grid


# ---------- ৪. প্রতিটা বাবলের darkness মেপে filled/blank ঠিক করা ----------
def read_block(gray, block, sample_radius=9, fill_gap_threshold=20):
    grid = bubble_grid(block)
    answers = []
    for q_idx, row in enumerate(grid):
        darks = []
        for (x, y) in row:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), sample_radius, 255, -1)
            mean_val = cv2.mean(gray, mask=mask)[0]
            darks.append(mean_val)

        darks = np.array(darks)
        sorted_idx = np.argsort(darks)
        darkest, second_darkest = darks[sorted_idx[0]], darks[sorted_idx[1]]
        gap = second_darkest - darkest

        options = ["A", "B", "C", "D", "E"][: block["options"]]

        if gap < fill_gap_threshold * 0.4:
            result, confidence = "BLANK", "high"
        elif gap < fill_gap_threshold:
            result, confidence = options[sorted_idx[0]], "low"
        else:
            result, confidence = options[sorted_idx[0]], "high"

        if gap < fill_gap_threshold and second_darkest < darks.mean() - fill_gap_threshold:
            result, confidence = "MULTIPLE", "low"

        answers.append({
            "qNo": q_idx + 1,
            "answer": result,
            "confidence": confidence,
            "raw": [round(d, 1) for d in darks],
        })
    return answers


# ---------- ৫. Roll Number গ্রিড পড়া ----------
def read_roll(gray, roll_block, sample_radius=9, fill_gap_threshold=20):
    x0, y0 = roll_block["first_bubble"]
    x1, y1 = roll_block["last_bubble"]
    n_digits = roll_block["digits"]
    n_values = 10

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


# ---------- ৬. মূল ফাংশন ----------
def read_omr_sheet(image_path, template_path):
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)

    img = cv2.imread(image_path)
    canon_size = tuple(template.get("image_size", CANON_SIZE))

    img, marker_found = warp_by_markers(img, canon_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    result = {
        "subjects": [], "needs_review": [],
        "roll": "", "roll_confidence": "high",
        "marker_aligned": marker_found,
    }

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
