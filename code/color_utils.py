import cv2
import numpy as np

def classify_team_color(cropped_img):
    if cropped_img is None or cropped_img.size == 0:
        return "Unknown"

    # Resize to small region for faster processing
    small_img = cv2.resize(cropped_img, (20, 20))
    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)

    # Flatten the image to a list of HSV pixels
    pixels = hsv.reshape(-1, 3)

    # Calculate mean color
    mean_hue = np.mean(pixels[:, 0])

    if 0 <= mean_hue <= 10 or 160 <= mean_hue <= 180:
        return "Red"
    elif 100 <= mean_hue <= 130:
        return "Blue"
    elif 20 <= mean_hue <= 40:
        return "Yellow"
    else:
        return "Unknown"
