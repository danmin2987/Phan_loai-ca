import cv2
import numpy as np
import math
# Read image

# use getHLStool to get these values
minL = 0
maxL = 80

minB = 20
maxB = 190

def hls_lthresh(roi, min_L, max_L):
    #min_L = 18
    #max_L = 150
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(roi, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(hls_l)
    #cv2.imshow('binary_output', binary_output)
    binary_output[(hls_l > min_L) & (hls_l <= max_L)] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def lab_bthresh(roi, min_B, max_B):
    #min_B = 40
    #max_B = 110
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > min_B) & (lab_b <= max_B))] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def getContour(binImg):
    # scale binary image to 0-255 image
    gray = (binImg*255).astype(np.uint8)
    
    # # Find Canny edges
    edged = cv2.Canny(gray, 30, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    dilated = cv2.dilate(edged, kernel)
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(roi, cnts, -1, (0, 255, 0), 1)
    return cnts, gray

cap = cv2.VideoCapture('/home/quangminh/Desktop/Plab /phan_loai_ca/phan_loai_ca/1978623541203566877.mp4')
while True:
    global binImg
    ret, frame = cap.read()
    x1, y1, x2, y2 = 115, 92, 640, 310
    roi = frame[y1:y2, x1:x2]
    #getContour(binImg)
# binImage  = hls_lthresh(img,minL,maxL)
    binImage = lab_bthresh(roi,minB,maxB)
    cnts = getContour(binImage)[0]
    gray = getContour(binImage)[1]
    for i in range(len(cnts)):
        rect = cv2.minAreaRect(cnts[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        l1 = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1]-box[1][1])**2)
        l2 = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1]-box[3][1])**2)
        print("[{}]Length-1: {:.02f} , Length-2: {:.02f}, Square: {:.02f}".format(i,l1,l2,l1*l2))
        
        if (l1*l2 >= 3000):
            cv2.drawContours(roi,[box],0,(0,0,255),2)

    cv2.imshow('Raw', roi)
    cv2.imshow('Raw1', gray)
    if cv2.waitKey(1) >= 0:
            break
cv2.destroyAllWindows()
