import cv2
import numpy as np

def detect_head_tail(roi, template_head, template_tail):
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Result1", gray_image)
    # Áp dụng template matching cho đầu cá
    result_head = cv2.matchTemplate(gray_image, template_head, cv2.TM_CCOEFF_NORMED)
    _, max_val_head, _, max_loc_head = cv2.minMaxLoc(result_head)
    
    # Áp dụng template matching cho đuôi cá
    result_tail = cv2.matchTemplate(gray_image, template_tail, cv2.TM_CCOEFF_NORMED)
    _, max_val_tail, _, max_loc_tail = cv2.minMaxLoc(result_tail)
    
    # Ngưỡng để xác định đầu và đuôi cá dựa trên kết quả tương đồng
    threshold = 0.65
    
    if max_val_head > threshold and max_val_tail > threshold:
        # Nếu cả đầu và đuôi có tương đồng đủ lớn
        if max_val_head > max_val_tail:
            x, y = max_loc_head
            print(x)
            w, h = template_head.shape[::-1]
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return "Đầu cá"
        else:
            x, y = max_loc_tail
            w, h = template_tail.shape[::-1]
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return "Đuôi cá"
    elif max_val_head > threshold:
        x, y = max_loc_head
        w, h = template_head.shape[::-1]
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return "Đầu cá"
    elif max_val_tail > threshold:
        x, y = max_loc_tail
        w, h = template_tail.shape[::-1]
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return "Đuôi cá"
    else:
        return "Không tìm thấy đầu hoặc đuôi cá"

# Đường dẫn đến hình ảnh và template đầu, đuôi cá
cap = cv2.VideoCapture('/home/quangminh/Desktop/Plab/phan_loai/phan_loai_ca/1978623541203566877.mp4')
while True:
    ret, frame = cap.read()
    x1, y1, x2, y2 = 115, 92, 640, 320
    roi = frame[y1:y2, x1:x2]
#image_path = '/home/quangminh/Desktop/Plab /phan_loai/phan_loai_ca/images.jpeg'
    template_head_path = '/home/quangminh/Desktop/Plab/phan_loai/phan_loai_ca/dau_.png'
    template_tail_path = '/home/quangminh/Desktop/Plab/phan_loai/phan_loai_ca/duoi_.png'

# Đọc hình ảnh và template
#image = cv2.imread(image_path)
    template_head = cv2.imread(template_head_path, 0)  # Chuyển đổi template sang ảnh grayscale
    template_tail = cv2.imread(template_tail_path, 0)

# Phát hiện đầu và đuôi cá
    result = detect_head_tail(roi, template_head, template_tail)
    print("Kết quả:", result)



# Hiển thị ảnh kết quả
    cv2.imshow("Result", roi)
    if cv2.waitKey(1) >= 0:
            break

#cv2.waitKey(0)
cv2.destroyAllWindows()
