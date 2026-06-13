from flask import Flask, request, render_template_string, send_file
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>6x6 Sudoku Solver</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body{ font-family:Arial; padding:20px; max-width:600px; margin:auto; }
        button{ padding:10px; margin-top:10px; }
        pre{ background:#f2f2f2; padding:10px; border-radius:8px; overflow-x:auto; }
        .error{ color:red; }
    </style>
</head>
<body>
<h2>6x6 Sudoku Solver (OCR)</h2>
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <br><br>
    <button type="submit">Solve</button>
</form>
{% if result %}
<hr>
<h3>Solution</h3>
<pre>{{ result }}</pre>
{% endif %}
{% if error %}
<p class="error">{{ error }}</p>
{% endif %}
</body>
</html>
"""

SIZE = 6

def is_valid(board, row, col, num):
    # row check
    for x in range(SIZE):
        if board[row][x] == num:
            return False
    # col check
    for x in range(SIZE):
        if board[x][col] == num:
            return False
    # 2x3 subgrid check
    start_row = (row // 2) * 2
    start_col = (col // 3) * 3
    for r in range(start_row, start_row + 2):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True

def solve(board):
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row][col] == 0:
                for num in range(1, 7):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def board_to_text(board):
    return "\n".join(" ".join(str(cell) for cell in row) for row in board)

def extract_sudoku_board(image_path):
    """ছবি থেকে 6x6 সুডোকু বোর্ড বের করে 2D লিস্ট আকারে রিটার্ন করে"""
    # ছবি পড়া এবং প্রিপ্রসেসিং
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ব্লার ও থ্রেশহোল্ড
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # কনট্যুর খোঁজা
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # সবচেয়ে বড় চারকোণা কনট্যুর বাছাই (সুডোকু গ্রিড)
    max_area = 0
    grid_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            grid_contour = approx

    if grid_contour is None:
        raise ValueError("ছবিতে সুডোকু গ্রিড খুঁজে পাওয়া যায়নি।")

    # পার্সপেক্টিভ ট্রান্সফর্ম করে সোজা করা
    pts = grid_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))

    # গ্রে স্কেল ও বাইনারী
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

    # 6x6 সেল ভাগ করা
    h, w = warped_thresh.shape
    cell_h = h // SIZE
    cell_w = w // SIZE

    board = [[0]*SIZE for _ in range(SIZE)]

    # প্রতিটি সেলে OCR - Use environment variable for Tesseract path
    tesseract_path = os.environ.get('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    for i in range(SIZE):
        for j in range(SIZE):
            y1 = i * cell_h
            y2 = (i+1) * cell_h
            x1 = j * cell_w
            x2 = (j+1) * cell_w

            cell_roi = warped_thresh[y1:y2, x1:x2]

            # সেলের ভিতরে সাদা অংশ বড় করলে ডিজিট ভালো পড়ে
            kernel = np.ones((2,2), np.uint8)
            cell_roi = cv2.dilate(cell_roi, kernel, iterations=1)

            # Tesseract কনফিগার করা - শুধু ডিজিট (1-6), সিঙ্গেল ক্যারেক্টার
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456'
            text = pytesseract.image_to_string(cell_roi, config=custom_config).strip()

            if text.isdigit():
                num = int(text)
                if 1 <= num <= 6:
                    board[i][j] = num
            # যদি খালি হয় (ফাঁকা সেল) তাহলে 0 ই থাকে

    return board

def order_points(pts):
    """কনট্যুরের চারটি পয়েন্টকে ক্রমান্বয়ে: টপ-লেফট, টপ-রাইট, বটম-রাইট, বটম-লেফট"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@app.route('/favicon.ico')
@app.route('/favicon.png')
def favicon():
    # Return a simple transparent 16x16 PNG as favicon
    img = Image.new('RGBA', (16, 16), (0, 0, 0, 0))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        image = request.files.get("image")
        if image and image.filename:
            os.makedirs("uploads", exist_ok=True)
            filepath = os.path.join("uploads", image.filename)
            image.save(filepath)

            try:
                # ছবি থেকে বোর্ড এক্সট্র্যাক্ট
                board = extract_sudoku_board(filepath)
                # বোর্ড কপি করে সমাধান
                board_copy = [row[:] for row in board]
                if solve(board_copy):
                    result = board_to_text(board_copy)
                else:
                    error = "দুঃখিত, এই সুডোকুর কোনো সমাধান নেই।"
            except Exception as e:
                error = f"প্রক্রিয়াকরণে ত্রুটি: {str(e)}"

    return render_template_string(HTML, result=result, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
