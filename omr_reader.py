import cv2
import json
import numpy as np

def align_sheet(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    sheet_contour = None
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 0.2 * img.shape[0] * img.shape[1]:
            sheet_contour = approx.reshape(4, 2)
            break

    if sheet_contour is None:
        return img

    pts = order_points(sheet_contour)
    (tl, tr, br, bl) = pts
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    return cv2.warpPerspective(img, M, (maxW, maxH))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

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

def read_block(gray, block, sample_radius=10, fill_gap_threshold=15):
    grid = bubble_grid(block)
    answers = []
    for q_idx, row in enumerate(grid):
        darks = []
        for (x, y) in row:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (int(round(x)), int(round(y))), sample_radius, 255, -1)
            darks.append(cv2.mean(gray, mask=mask)[0])

        darks = np.array(darks)
        sorted_idx = np.argsort(darks)
        gap = darks[sorted_idx[1]] - darks[sorted_idx[0]]
        options = ["A", "B", "C", "D", "E"][: block["options"]]

        if gap < fill_gap_threshold * 0.4:
            result, confidence = "BLANK", "high"
        elif gap < fill_gap_threshold:
            result, confidence = options[sorted_idx[0]], "low"
        else:
            result, confidence = options[sorted_idx[0]], "high"

        answers.append({
            "qNo": q_idx + 1,
            "answer": result,
            "confidence": confidence
        })
    return answers

def read_roll(gray, roll_block, sample_radius=10, fill_gap_threshold=15):
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
            cv2.circle(mask, (int(round(x)), int(round(y))), sample_radius, 255, -1)
            darks.append(cv2.mean(gray, mask=mask)[0])
        
        darks = np.array(darks)
        order = np.argsort(darks)
        gap = darks[order[1]] - darks[order[0]]
        digit = str(order[0]) if gap > fill_gap_threshold else "?"
        if gap <= fill_gap_threshold:
            low_confidence_cols.append(col + 1)
        digits.append(digit)

    return "".join(digits), low_confidence_cols

def read_omr_sheet(image_path, template_path):
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)

    img = cv2.imread(image_path)
    img = align_sheet(img)

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
