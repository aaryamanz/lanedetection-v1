import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])# bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines
 
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny
 
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image
 
def region_of_interest(image):
    height = image.shape[0]  # Image height
    width = image.shape[1]   # Image width

    polygons = np.array([
        [(int(width * 0.1), height),  # Bottom left, adjusted to capture the left white line
        (int(width * 0.9), height),  # Bottom right, adjusted to capture the right white line
        (int(width * 0.55), int(height * 0.6)),  # Top right, slightly to the left of the center
        (int(width * 0.45), int(height * 0.6))]  # Top left, slightly to the right of the center
    ], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
 
 
def process_frame(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start_time = time.time()
    outs = net.forward(output_layers)
    elapsed_time = time.time() - start_time
    return outs, elapsed_time, height, width

def calculate_curvature_and_position(lines, width):
    if lines is None:
        return None, None

    # Fit polynomial to lane lines, assuming the form: y = Ax^2 + Bx + C
    try:
        left_x = [line[0] for line in lines[0]]
        left_y = [line[1] for line in lines[0]]
        right_x = [line[2] for line in lines[1]]
        right_y = [line[3] for line in lines[1]]
        
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        
        # Calculate the radius of curvature in pixels
        y_eval = np.max(left_y + right_y)
        left_curvature = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curvature = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        # Calculate vehicle position with respect to center
        left_lane_bottom_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_lane_bottom_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_lane_bottom_x + right_lane_bottom_x) / 2
        vehicle_position = width / 2
        center_offset = (vehicle_position - lane_center) * 3.7 / 700 # assuming lane width = 3.7 meters

        return (left_curvature + right_curvature) / 2, center_offset
    except TypeError:
        return None, None

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# lane_canny = canny(lane_image)
# cropped_canny = region_of_interest(lane_canny)
# lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
# averaged_lines = average_slope_intercept(image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)

cap = cv2.VideoCapture("test3.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_1 = canny(frame)
    cropped_image = region_of_interest(canny_1)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, None, minLineLength= 40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines) #we don't average out lines rn
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
        
    outs, elapsed_time, height, width = process_frame(frame, net, output_layers)
     # Implement lane detection, averaging, and line drawing as before, now including curvature calculation
    curvature, position_offset = calculate_curvature_and_position(averaged_lines, width)

    # Display the results
    info = [
        f"Time per Frame: {elapsed_time:.3f} sec",
        f"Frame Rate: {1/elapsed_time:.2f} FPS",
        f"Curvature: {curvature:.2f} m",
        f"Position Offset: {position_offset:.2f} m from center"
    ]
    
    # Information to show on the screen (class names and boxes)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(combo_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(combo_image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    cv2.imshow("results", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()