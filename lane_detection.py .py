import cv2
import numpy as np
import os

# ============================================================
# PARAMETERS
# ============================================================
CANNY_LOW_THRESHOLD  = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_RHO            = 2
HOUGH_THETA          = np.pi / 180
HOUGH_THRESHOLD      = 30
HOUGH_MIN_LINE_LEN   = 20
HOUGH_MAX_LINE_GAP   = 1

# Lane width tính theo % chiều rộng ảnh (thay vì pixel cố định)
# Bottom (gần camera): rộng hơn — 22% width
# Top (xa camera):     hẹp hơn  — 8% width  (perspective effect)
LANE_WIDTH_RATIO_BOTTOM = 0.22
LANE_WIDTH_RATIO_TOP    = 0.08

INPUT_DIR  = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: HSV Color Isolation — chỉ giữ TRẮNG và VÀNG
# ============================================================
def isolate_lane_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15,  80,  80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0,   0,  200])
    upper_white = np.array([255, 30, 255])
    white_mask  = cv2.inRange(hsv, lower_white, upper_white)

    combined = cv2.bitwise_or(yellow_mask, white_mask)
    return cv2.bitwise_and(image, image, mask=combined)

# ============================================================
# STEP 2: Grayscale + Gaussian Blur
# ============================================================
def preprocess(color_isolated):
    gray = cv2.cvtColor(color_isolated, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)

# ============================================================
# STEP 3: Canny Edge Detection
# ============================================================
def detect_edges(blurred):
    return cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

# ============================================================
# STEP 4: ROI Mask
# ============================================================
def region_of_interest(edges, shape):
    h, w = shape[:2]
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (0,        h),
        (w * 0.35, h * 0.60),
        (w * 0.65, h * 0.60),
        (w,        h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)

# ============================================================
# STEP 5: Hough Transform
# ============================================================
def hough_lines(masked_edges):
    return cv2.HoughLinesP(
        masked_edges,
        rho           = HOUGH_RHO,
        theta         = HOUGH_THETA,
        threshold     = HOUGH_THRESHOLD,
        minLineLength = HOUGH_MIN_LINE_LEN,
        maxLineGap    = HOUGH_MAX_LINE_GAP,
    )

# ============================================================
# STEP 6: Median slope/intercept (từ Medium article)
# ============================================================
def compute_lane_equations(lines):
    if lines is None:
        return None, None, None, None

    mxb = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        mxb.append([m, b])

    if not mxb:
        return None, None, None, None

    mxb = np.array(mxb)
    left_mask  = mxb[:, 0] < -0.3
    right_mask = mxb[:, 0] >  0.3

    left_m  = np.median(mxb[left_mask,  0]) if left_mask.any()  else None
    left_b  = np.median(mxb[left_mask,  1]) if left_mask.any()  else None
    right_m = np.median(mxb[right_mask, 0]) if right_mask.any() else None
    right_b = np.median(mxb[right_mask, 1]) if right_mask.any() else None

    return left_m, left_b, right_m, right_b

# ============================================================
# STEP 7: Tính toạ độ đường từ slope & intercept
# ============================================================
def calc_line_coords(m, b, y_bottom, y_top):
    if m is None or b is None or np.isnan(m) or np.isnan(b):
        return None
    try:
        x_bottom = int((y_bottom - b) / m)
        x_top    = int((y_top    - b) / m)
        return (x_bottom, y_bottom, x_top, y_top)
    except:
        return None

# ============================================================
# STEP 8: Estimate làn bị thiếu — PERSPECTIVE-AWARE
# Bottom rộng hơn top (giống hình thang hội tụ về phía xa)
# ============================================================
def estimate_missing_lane(left_coords, right_coords, image_width):
    left_estimated  = False
    right_estimated = False

    # Width thay đổi theo perspective
    w_bottom = int(image_width * LANE_WIDTH_RATIO_BOTTOM)  # rộng ở gần
    w_top    = int(image_width * LANE_WIDTH_RATIO_TOP)      # hẹp ở xa

    # Có right, thiếu left → dịch sang TRÁI
    if right_coords is not None and left_coords is None:
        left_coords = (
            right_coords[0] - w_bottom,  # bottom dịch nhiều
            right_coords[1],
            right_coords[2] - w_top,     # top dịch ít (perspective)
            right_coords[3],
        )
        left_estimated = True

    # Có left, thiếu right → dịch sang PHẢI
    elif left_coords is not None and right_coords is None:
        right_coords = (
            left_coords[0] + w_bottom,   # bottom dịch nhiều
            left_coords[1],
            left_coords[2] + w_top,      # top dịch ít (perspective)
            left_coords[3],
        )
        right_estimated = True

    return left_coords, right_coords, left_estimated, right_estimated

# ============================================================
# STEP 9: Vẽ làn + polygon + CENTER POINT
# ============================================================
def draw_lanes(image, left_coords, right_coords,
               left_estimated=False, right_estimated=False):
    overlay = np.zeros_like(image)

    # Left lane — xanh dương (thật) / cam (estimated)
    if left_coords is not None:
        color = (0, 140, 255) if left_estimated else (255, 0, 0)
        cv2.line(overlay,
                 (left_coords[0],  left_coords[1]),
                 (left_coords[2],  left_coords[3]),
                 color, 10)
        if left_estimated:
            mx = (left_coords[0] + left_coords[2]) // 2
            my = (left_coords[1] + left_coords[3]) // 2
            cv2.putText(overlay, "estimated", (mx - 50, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)

    # Right lane — đỏ (thật) / cam (estimated)
    if right_coords is not None:
        color = (0, 140, 255) if right_estimated else (0, 0, 255)
        cv2.line(overlay,
                 (right_coords[0], right_coords[1]),
                 (right_coords[2], right_coords[3]),
                 color, 10)
        if right_estimated:
            mx = (right_coords[0] + right_coords[2]) // 2
            my = (right_coords[1] + right_coords[3]) // 2
            cv2.putText(overlay, "estimated", (mx - 50, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)

    # Polygon + Center khi có đủ 2 làn
    if left_coords is not None and right_coords is not None:

        poly_color = (0, 180, 0) if not (left_estimated or right_estimated) else (0, 140, 0)
        pts = np.array([
            [left_coords[0],  left_coords[1]],
            [left_coords[2],  left_coords[3]],
            [right_coords[2], right_coords[3]],
            [right_coords[0], right_coords[1]],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], poly_color)

        # ── CENTER POINT (Extra Credit) ───────────────────────
        cx_bottom = (left_coords[0] + right_coords[0]) // 2
        cy_bottom = (left_coords[1] + right_coords[1]) // 2
        cx_top    = (left_coords[2] + right_coords[2]) // 2
        cy_top    = (left_coords[3] + right_coords[3]) // 2

        cv2.line(overlay, (cx_bottom, cy_bottom), (cx_top, cy_top), (0, 255, 255), 3)
        cv2.circle(overlay, (cx_bottom, cy_bottom), 20, (0, 255, 0),    -1)
        cv2.circle(overlay, (cx_bottom, cy_bottom), 24, (255, 255, 255),  3)
        cv2.putText(overlay, "Center",
                    (cx_bottom - 40, cy_bottom - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        # ──────────────────────────────────────────────────────

    return cv2.addWeighted(image, 0.8, overlay, 0.6, 0)

# ============================================================
# FULL PIPELINE cho 1 ảnh
# ============================================================
def detect_lanes(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIP] Không đọc được: {image_path}")
        return

    h, w     = image.shape[:2]
    y_bottom = h
    y_top    = int(h * 0.60)

    color_isolated = isolate_lane_colors(image)
    blurred        = preprocess(color_isolated)
    edges          = detect_edges(blurred)
    masked         = region_of_interest(edges, image.shape)
    lines          = hough_lines(masked)

    left_m, left_b, right_m, right_b = compute_lane_equations(lines)

    left_coords  = calc_line_coords(left_m,  left_b,  y_bottom, y_top)
    right_coords = calc_line_coords(right_m, right_b, y_bottom, y_top)

    # Estimate với perspective-aware width
    left_coords, right_coords, left_est, right_est = estimate_missing_lane(
        left_coords, right_coords, w
    )

    result = draw_lanes(image, left_coords, right_coords, left_est, right_est)
    cv2.imwrite(output_path, result)

    found = []
    if left_coords:  found.append("left (est)" if left_est  else "left ✓")
    if right_coords: found.append("right (est)" if right_est else "right ✓")
    if left_coords and right_coords: found.append("center ✓")
    status = ", ".join(found) if found else "không detect được"
    print(f"[OK] {os.path.basename(image_path):15s} → {os.path.basename(output_path):16s}  |  {status}")

# ============================================================
# Xử lý tất cả 10 ảnh (.png)
# ============================================================
print("=" * 65)
print("  Lane Detection — 10 images  (HSV + Median + Perspective)")
print("=" * 65)

for i in range(1, 11):
    detect_lanes(
        os.path.join(INPUT_DIR,  f"image{i}.png"),
        os.path.join(OUTPUT_DIR, f"output{i}.png"),
    )

print("=" * 65)
print(f"  Xong! Kết quả lưu tại '{OUTPUT_DIR}/'")