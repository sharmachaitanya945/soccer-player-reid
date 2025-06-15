import cv2
import numpy as np

def classify_team_color(cropped_img):
    """Enhanced color classification function with better color detection"""
    if cropped_img is None or cropped_img.size == 0:
        return "Unknown"
    
    # Resize to small region for faster processing
    small_img = cv2.resize(cropped_img, (30, 30))
    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    
    # Focus on the center region (jersey area) - avoid edges which might have grass/background
    center_h, center_w = hsv.shape[:2]
    center_region = hsv[center_h//3:2*center_h//3, center_w//3:2*center_w//3]
    
    # Flatten the image to a list of HSV pixels
    pixels = center_region.reshape(-1, 3)
    
    # Filter out low saturation pixels (white/gray areas like shorts, socks, etc.)
    high_sat_pixels = pixels[pixels[:, 1] > 30]  # Saturation > 30
    
    if len(high_sat_pixels) == 0:
        # If no high saturation pixels, check for white/light colors
        light_pixels = pixels[pixels[:, 2] > 200]  # Value > 200 (bright)
        if len(light_pixels) > len(pixels) * 0.3:  # If 30% or more pixels are bright
            return "White"
        return "Unknown"
    
    # Calculate mean hue and saturation of high saturation pixels
    mean_hue = np.mean(high_sat_pixels[:, 0])
    mean_sat = np.mean(high_sat_pixels[:, 1])
    mean_val = np.mean(high_sat_pixels[:, 2])
    
    # Enhanced color classification with better ranges
    if (0 <= mean_hue <= 12 or 168 <= mean_hue <= 180) and mean_sat > 50:
        return "Red"
    elif 90 <= mean_hue <= 130 and mean_sat > 50:
        return "Blue"
    elif 15 <= mean_hue <= 35 and mean_sat > 60:
        return "Yellow"
    elif 35 <= mean_hue <= 85 and mean_sat > 50:
        return "Green"
    elif 130 <= mean_hue <= 168 and mean_sat > 50:
        return "Purple"
    elif mean_val < 60:  # Dark colors
        return "Black"
    elif mean_sat < 40 and mean_val > 180:  # Light, low saturation
        return "White"
    else:
        return "Unknown"


def determine_team_from_color(detected_color, team1_color, team2_color):
    """
    Determine which team a player belongs to based on detected color and team configuration
    """
    # Direct name matching (case-insensitive)
    if detected_color.lower() == team1_color["name"].lower():
        return "team1"
    elif detected_color.lower() == team2_color["name"].lower():
        return "team2"
    
    # Color similarity matching for common color variations
    color_variants = {
        "red": ["red", "crimson", "maroon"],
        "blue": ["blue", "navy", "royal", "sky"],
        "green": ["green", "forest", "lime"],
        "yellow": ["yellow", "gold", "amber"],
        "white": ["white", "silver", "light"],
        "black": ["black", "dark"],
        "purple": ["purple", "violet", "magenta"]
    }
    
    # Check if detected color matches any variant of team colors
    for base_color, variants in color_variants.items():
        if detected_color.lower() in variants:
            if team1_color["name"].lower() in variants:
                return "team1"
            elif team2_color["name"].lower() in variants:
                return "team2"
    
    return "unknown"


def test_color_detection_on_crop(cropped_image):
    """
    Test function to debug color detection on a specific crop
    """
    detected_color = classify_team_color(cropped_image)
    print(f"Detected color: {detected_color}")
    
    # Show the cropped image for visual inspection
    cv2.imshow("Cropped Player", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected_color