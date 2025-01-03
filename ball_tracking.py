import cv2
import numpy as np

# Capture the video
vid = cv2.VideoCapture(r'D:\prisha_manipal_sp\sp_rugved\Ball_Tracking.mp4')

while True:
    ret, frame = vid.read()
    if not ret:
        break  
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([38, 78, 52])    
    upper_color = np.array([66, 140, 108])
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, "ball",(int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('ball tracking', frame)
    
    if cv2.waitKey(15) & 0xFF == ord('d'):
        break

vid.release()
cv2.destroyAllWindows()

#look up object tracking 