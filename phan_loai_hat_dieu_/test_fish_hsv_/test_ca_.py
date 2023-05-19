import cv2
import numpy as np
import multiprocessing

from multiprocessing import Queue
queue = Queue()

def c1():
    cap = cv2.VideoCapture('/home/quangminh/Videos/smart_lightting-/video_so-che_ca_/5543479303214953217.mp4')
    while True:
        ret, frame = cap.read()
        queue.put(frame)
        cv2.imshow('video',frame)
        x1, y1, x2, y2 = 400,85,550,300
        frame1 = frame[y1:y2, x1:x2] 
        #cv2.imshow('video1',frame1)
        hsv_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

        conveyor_belt_lower = np.array([90, 50, 50])
        conveyor_belt_upper = np.array([130, 255, 255])
        fish_lower = np.array([0, 0, 200])
        fish_upper = np.array([180, 30, 255])

    # Tạo mask cho băng tải và cá dựa trên khoảng giá trị màu đã định nghĩa
        conveyor_belt_mask = cv2.inRange(hsv_frame, conveyor_belt_lower, conveyor_belt_upper)
        fish_mask = cv2.inRange(hsv_frame, fish_lower, fish_upper)

    # Đảo ngược mask của băng tải để lấy mask cho phần còn lại của hình ảnh (các con cá)
        conveyor_belt_mask_inv = cv2.bitwise_not(conveyor_belt_mask)

    # Áp dụng mask của băng tải vào ảnh gốc để lấy phần băng tải
        conveyor_belt = cv2.bitwise_and(frame1, frame1, mask=conveyor_belt_mask_inv)

    # Áp dụng mask của cá vào ảnh gốc để lấy phần cá
        fish = cv2.bitwise_and(frame1, frame1, mask=fish_mask)

    # Chuyển phần băng tải sang ảnh đen trắng và áp dụng ngưỡng để tạo thành mask nhị phân
        conveyor_belt_gray = cv2.cvtColor(conveyor_belt, cv2.COLOR_BGR2GRAY)
        ret, conveyor_belt_thresh = cv2.threshold(conveyor_belt_gray, 10, 255, cv2.THRESH_BINARY)

    # Đảo ngược mask nhị phân của băng tải để lấy mask cho phần còn lại của hình ảnh (các con cá)
        conveyor_belt_thresh_inv = cv2.bitwise_not(conveyor_belt_thresh)

    # Áp dụng mask nhị phân của băng tải vào phần cá để làm nền trắng cho cá
        fish_white_bg = cv2.bitwise_and(fish, fish, mask=conveyor_belt_thresh_inv)

    # Áp dụng mask nhị phân của băng tải vào phần băng tải để làm nền đen cho băng tải
        conveyor_belt_black_bg = cv2.bitwise_and(conveyor_belt, conveyor_belt, mask=conveyor_belt_thresh)

    # Kết hợp phần cá với nền trắng và phần băng tải với nền đen
        result = cv2.add(fish_white_bg, conveyor_belt_black_bg)
        pixel_size_sq = 0.01 # Kích thước mỗi pixel (đơn vị mm^2)
        fish_area = cv2.countNonZero(fish_mask) * pixel_size_sq
        print(fish_area)
        cv2.imshow('video3',result)
        if fish_area < 200:
        # Nếu diện tích lớn hơn ngưỡng, đó là một con cá
        # Phân loại các con cá dựa trên diện tích của chúng
                if fish_area > 5:
                    print('a')
        if cv2.waitKey(1) >= 0:
            break
                
def c2():
    
    while True:
        frame = queue.get()
        x1, y1, x2, y2 = 150,85,300,300
        frame2 = frame[y1:y2, x1:x2] 
        #cv2.imshow('video2',frame2)
        hsv_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        conveyor_belt_lower = np.array([90, 50, 50])
        conveyor_belt_upper = np.array([130, 255, 255])
        fish_lower = np.array([0, 0, 200])
        fish_upper = np.array([180, 30, 255])

    # Tạo mask cho băng tải và cá dựa trên khoảng giá trị màu đã định nghĩa
        conveyor_belt_mask = cv2.inRange(hsv_frame, conveyor_belt_lower, conveyor_belt_upper)
        fish_mask = cv2.inRange(hsv_frame, fish_lower, fish_upper)

    # Đảo ngược mask của băng tải để lấy mask cho phần còn lại của hình ảnh (các con cá)
        conveyor_belt_mask_inv = cv2.bitwise_not(conveyor_belt_mask)

    # Áp dụng mask của băng tải vào ảnh gốc để lấy phần băng tải
        conveyor_belt = cv2.bitwise_and(frame2, frame2, mask=conveyor_belt_mask_inv)

    # Áp dụng mask của cá vào ảnh gốc để lấy phần cá
        fish = cv2.bitwise_and(frame2, frame2, mask=fish_mask)

    # Chuyển phần băng tải sang ảnh đen trắng và áp dụng ngưỡng để tạo thành mask nhị phân
        conveyor_belt_gray = cv2.cvtColor(conveyor_belt, cv2.COLOR_BGR2GRAY)
        ret, conveyor_belt_thresh = cv2.threshold(conveyor_belt_gray, 10, 255, cv2.THRESH_BINARY)

    # Đảo ngược mask nhị phân của băng tải để lấy mask cho phần còn lại của hình ảnh (các con cá)
        conveyor_belt_thresh_inv = cv2.bitwise_not(conveyor_belt_thresh)

    # Áp dụng mask nhị phân của băng tải vào phần cá để làm nền trắng cho cá
        fish_white_bg = cv2.bitwise_and(fish, fish, mask=conveyor_belt_thresh_inv)

    # Áp dụng mask nhị phân của băng tải vào phần băng tải để làm nền đen cho băng tải
        conveyor_belt_black_bg = cv2.bitwise_and(conveyor_belt, conveyor_belt, mask=conveyor_belt_thresh)

    # Kết hợp phần cá với nền trắng và phần băng tải với nền đen
        result = cv2.add(fish_white_bg, conveyor_belt_black_bg)
        pixel_size_sq = 0.01 # Kích thước mỗi pixel (đơn vị mm^2)
        fish_area = cv2.countNonZero(fish_mask) * pixel_size_sq
        #print(fish_area)
        cv2.imshow('video4',result)
        if fish_area < 200:
        # Nếu diện tích lớn hơn ngưỡng, đó là một con cá
        # Phân loại các con cá dựa trên diện tích của chúng
                if fish_area > 2:
                    print('b')
        if cv2.waitKey(1) >= 0:
            break 
def c3():
    
    while True:
        frame = queue.get()
        x1, y1, x2, y2 = 560,85,700,300
        frame3 = frame[y1:y2, x1:x2] 
        #cv2.imshow('video5',frame3)
        hsv_frame = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)

        conveyor_belt_lower = np.array([90, 50, 50])
        conveyor_belt_upper = np.array([130, 255, 255])
        fish_lower = np.array([0, 0, 200])
        fish_upper = np.array([180, 30, 255])

    # Tạo mask cho băng tải và cá dựa trên khoảng giá trị màu đã định nghĩa
        conveyor_belt_mask = cv2.inRange(hsv_frame, conveyor_belt_lower, conveyor_belt_upper)
        fish_mask = cv2.inRange(hsv_frame, fish_lower, fish_upper)

    # Đảo ngược mask của băng tải để lấy mask cho phần còn lại của hình ảnh (các con cá)
        conveyor_belt_mask_inv = cv2.bitwise_not(conveyor_belt_mask)

    # Áp dụng mask của băng tải vào ảnh gốc để lấy phần băng tải
        conveyor_belt = cv2.bitwise_and(frame3, frame3, mask=conveyor_belt_mask_inv)

    # Áp dụng mask của cá vào ảnh gốc để lấy phần cá
        fish = cv2.bitwise_and(frame3, frame3, mask=fish_mask)

    # Chuyển phần băng tải sang ảnh đen trắng và áp dụng ngưỡng để tạo thành mask nhị phân
        conveyor_belt_gray = cv2.cvtColor(conveyor_belt, cv2.COLOR_BGR2GRAY)
        ret, conveyor_belt_thresh = cv2.threshold(conveyor_belt_gray, 10, 255, cv2.THRESH_BINARY)

    # Đảo ngược mask nhị phân của băng tải để lấy mask cho phần còn lại của hình ảnh (các con cá)
        conveyor_belt_thresh_inv = cv2.bitwise_not(conveyor_belt_thresh)

    # Áp dụng mask nhị phân của băng tải vào phần cá để làm nền trắng cho cá
        fish_white_bg = cv2.bitwise_and(fish, fish, mask=conveyor_belt_thresh_inv)

    # Áp dụng mask nhị phân của băng tải vào phần băng tải để làm nền đen cho băng tải
        conveyor_belt_black_bg = cv2.bitwise_and(conveyor_belt, conveyor_belt, mask=conveyor_belt_thresh)

    # Kết hợp phần cá với nền trắng và phần băng tải với nền đen
        result = cv2.add(fish_white_bg, conveyor_belt_black_bg)
        pixel_size_sq = 0.01 # Kích thước mỗi pixel (đơn vị mm^2)
        fish_area = cv2.countNonZero(fish_mask) * pixel_size_sq
        #print(fish_area)
        cv2.imshow('video6',result)
        if fish_area < 200:
        # Nếu diện tích lớn hơn ngưỡng, đó là một con cá
        # Phân loại các con cá dựa trên diện tích của chúng
                if fish_area > 1:
                    print('c')
        if cv2.waitKey(1) >= 0:
            break        
        #cv2.imshow('video2',result)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
tCap1 = multiprocessing.Process(target=c1)
tCap2 = multiprocessing.Process(target=c2)
tCap3 = multiprocessing.Process(target=c3)
tCap1.start()
tCap2.start()
tCap3.start()
tCap1.join()
tCap2.join()
tCap3.join()
