import cv2
import numpy as np

def detect_fire(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the "fire" color
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the image to get the "fire" mask
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Define the lower and upper bounds for the "fire" color (again)
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])

    # Threshold the image to get the "fire" mask (again)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply a median blur to reduce noise
    mask = cv2.medianBlur(mask, 5)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate the total area of the fire-like regions
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        total_area += area


    if(len(contours) == 0):
        score = 0
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        _,_,w,h = cv2.boundingRect(largest_contour)
        score = total_area / (w*h)

    # Draw bounding boxes around the fire-like pixels
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)


    cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)

    # Show the image with the boxes and the prediction score
    cv2.putText(image, "Fire score: {:.2f}".format(score), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread('./images/dronetfire2.png')

# Call the detect_fire function
detect_fire(image)
