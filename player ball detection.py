import cv2
import numpy as np

def detect_ball_and_player(img):
    #split channels
    b, g, r    = cv2.split(img)

    #remove ground
    g_greaterthan_r = np.greater(g,r)
    g_greaterthan_b = np.greater(g,b)
    ground_removed = np.logical_and(g_greaterthan_r, g_greaterthan_b)

    #canny edge detection
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,250)

    #binary or operation between ground_removed image and edge detection
    binary_or = np.logical_or(ground_removed, edges)

    #inverse 0 and 1 to make foreground objects white (1) and background black(0)
    binary_not = np.logical_not(binary_or).astype(np.uint8)

    #morphological operation 
    kernel = np.ones((9,9),np.uint8)
    result = cv2.morphologyEx(binary_not, cv2.MORPH_CLOSE, kernel)

    #find connected components
    connected_components = cv2.connectedComponentsWithStats(result, 4, cv2.CV_32S)
    (num_components, component_ids, values, centroid) = connected_components

    #filter out useful components by area
    for i in range(1, num_components):
        area = values[i, cv2.CC_STAT_AREA]
        left = values[i, cv2.CC_STAT_LEFT] 
        top = values[i, cv2.CC_STAT_TOP] 
        width = values[i, cv2.CC_STAT_WIDTH] 
        height = values[i, cv2.CC_STAT_HEIGHT]     

        # mark ball
        if area>40 and area <150:
            cv2.rectangle(img,(left,top),(left+width,top+height),(0,0,255),3)
        
        # mark player
        if area>200 and area <2500:
            cv2.rectangle(img,(left,top),(left+width,top+height),(0,255,255),3)

    return img

#### read a soccer video and process frame by frame

cap = cv2.VideoCapture('F://1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    #detect the ball and players in the frame
    out = detect_ball_and_player(frame)

    #display result
    cv2.imshow('Ball and player detection', out)
    
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()