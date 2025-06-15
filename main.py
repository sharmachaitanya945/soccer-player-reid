import sys
import cv2
import torch
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
                           QFileDialog, QHBoxLayout, QFrame, QStatusBar, QProgressBar,
                           QLineEdit, QComboBox, QColorDialog, QDialog, QGridLayout,
                           QGroupBox, QFormLayout, QMenuBar, QAction)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import QTimer, Qt
from ultralytics import YOLO


def classify_team_color(cropped_img):
    """Improved color classification function"""
    if cropped_img is None or cropped_img.size == 0:
        return "Unknown"
    
    # Resize to small region for faster processing
    small_img = cv2.resize(cropped_img, (30, 30))
    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    
    # Focus on the center region (jersey area)
    center_h, center_w = hsv.shape[:2]
    center_region = hsv[center_h//4:3*center_h//4, center_w//4:3*center_w//4]
    
    # Flatten the image to a list of HSV pixels
    pixels = center_region.reshape(-1, 3)
    
    # Filter out low saturation pixels (white/gray areas)
    high_sat_pixels = pixels[pixels[:, 1] > 50]  # Saturation > 50
    
    if len(high_sat_pixels) == 0:
        return "Unknown"
    
    # Calculate mean hue of high saturation pixels
    mean_hue = np.mean(high_sat_pixels[:, 0])
    mean_sat = np.mean(high_sat_pixels[:, 1])
    mean_val = np.mean(high_sat_pixels[:, 2])
    
    # Improved color classification
    if (0 <= mean_hue <= 15 or 165 <= mean_hue <= 180) and mean_sat > 80:
        return "Red"
    elif 90 <= mean_hue <= 130 and mean_sat > 80:
        return "Blue"
    elif 15 <= mean_hue <= 35 and mean_sat > 80:
        return "Yellow"
    elif 35 <= mean_hue <= 85 and mean_sat > 80:
        return "Green"
    elif 130 <= mean_hue <= 165 and mean_sat > 80:
        return "Purple"
    elif mean_val < 50:  # Dark colors
        return "Black"
    elif mean_sat < 30 and mean_val > 200:  # Light, low saturation
        return "White"
    else:
        return "Unknown"


class TeamConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Team Configuration")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Team 1 Group
        team1_group = QGroupBox("Team 1")
        team1_layout = QFormLayout()
        
        self.team1_name = QLineEdit("Team 1")
        team1_layout.addRow("Name:", self.team1_name)
        
        self.team1_color_combo = QComboBox()
        self.team1_color_combo.addItems(["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Custom..."])
        self.team1_color_combo.setCurrentText("Red")
        self.team1_color_combo.currentTextChanged.connect(self.on_team1_color_changed)
        team1_layout.addRow("Color:", self.team1_color_combo)
        
        self.team1_custom_color = QColor(255, 0, 0)  # Default red
        self.team1_color_btn = QPushButton()
        self.team1_color_btn.setFixedSize(24, 24)
        self.team1_color_btn.setStyleSheet(f"background-color: rgb(255, 0, 0); border: none;")
        self.team1_color_btn.clicked.connect(self.pick_team1_color)
        team1_layout.addRow("Custom color:", self.team1_color_btn)
        
        team1_group.setLayout(team1_layout)
        layout.addWidget(team1_group)
        
        # Team 2 Group
        team2_group = QGroupBox("Team 2")
        team2_layout = QFormLayout()
        
        self.team2_name = QLineEdit("Team 2")
        team2_layout.addRow("Name:", self.team2_name)
        
        self.team2_color_combo = QComboBox()
        self.team2_color_combo.addItems(["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Custom..."])
        self.team2_color_combo.setCurrentText("Blue")
        self.team2_color_combo.currentTextChanged.connect(self.on_team2_color_changed)
        team2_layout.addRow("Color:", self.team2_color_combo)
        
        self.team2_custom_color = QColor(0, 0, 255)  # Default blue
        self.team2_color_btn = QPushButton()
        self.team2_color_btn.setFixedSize(24, 24)
        self.team2_color_btn.setStyleSheet(f"background-color: rgb(0, 0, 255); border: none;")
        self.team2_color_btn.clicked.connect(self.pick_team2_color)
        team2_layout.addRow("Custom color:", self.team2_color_btn)
        
        team2_group.setLayout(team2_layout)
        layout.addWidget(team2_group)
        
        # Referee settings
        referee_group = QGroupBox("Referee")
        referee_layout = QFormLayout()
        
        self.referee_color_combo = QComboBox()
        self.referee_color_combo.addItems(["Yellow", "Black", "White", "Custom..."])
        self.referee_color_combo.setCurrentText("Yellow")
        self.referee_color_combo.currentTextChanged.connect(self.on_referee_color_changed)
        referee_layout.addRow("Color:", self.referee_color_combo)
        
        self.referee_custom_color = QColor(255, 255, 0)  # Default yellow
        self.referee_color_btn = QPushButton()
        self.referee_color_btn.setFixedSize(24, 24)
        self.referee_color_btn.setStyleSheet(f"background-color: rgb(255, 255, 0); border: none;")
        self.referee_color_btn.clicked.connect(self.pick_referee_color)
        referee_layout.addRow("Custom color:", self.referee_color_btn)
        
        referee_group.setLayout(referee_layout)
        layout.addWidget(referee_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Hide custom color buttons initially
        self.update_color_button_visibility()
        
    def update_color_button_visibility(self):
        self.team1_color_btn.setVisible(self.team1_color_combo.currentText() == "Custom...")
        self.team2_color_btn.setVisible(self.team2_color_combo.currentText() == "Custom...")
        self.referee_color_btn.setVisible(self.referee_color_combo.currentText() == "Custom...")
    
    def on_team1_color_changed(self, color_name):
        self.update_color_button_visibility()
        if color_name == "Custom...":
            self.pick_team1_color()
    
    def on_team2_color_changed(self, color_name):
        self.update_color_button_visibility()
        if color_name == "Custom...":
            self.pick_team2_color()
    
    def on_referee_color_changed(self, color_name):
        self.update_color_button_visibility()
        if color_name == "Custom...":
            self.pick_referee_color()
    
    def pick_team1_color(self):
        color = QColorDialog.getColor(self.team1_custom_color, self, "Select Team 1 Color")
        if color.isValid():
            self.team1_custom_color = color
            self.team1_color_btn.setStyleSheet(f"background-color: {color.name()}; border: none;")
    
    def pick_team2_color(self):
        color = QColorDialog.getColor(self.team2_custom_color, self, "Select Team 2 Color")
        if color.isValid():
            self.team2_custom_color = color
            self.team2_color_btn.setStyleSheet(f"background-color: {color.name()}; border: none;")
    
    def pick_referee_color(self):
        color = QColorDialog.getColor(self.referee_custom_color, self, "Select Referee Color")
        if color.isValid():
            self.referee_custom_color = color
            self.referee_color_btn.setStyleSheet(f"background-color: {color.name()}; border: none;")
    
    def get_team1_color_data(self):
        if self.team1_color_combo.currentText() == "Custom...":
            return {"name": "Custom", "rgb": (self.team1_custom_color.red(), 
                                            self.team1_custom_color.green(), 
                                            self.team1_custom_color.blue())}
        return {"name": self.team1_color_combo.currentText(), "rgb": self.get_rgb_for_color(self.team1_color_combo.currentText())}
    
    def get_team2_color_data(self):
        if self.team2_color_combo.currentText() == "Custom...":
            return {"name": "Custom", "rgb": (self.team2_custom_color.red(), 
                                            self.team2_custom_color.green(), 
                                            self.team2_custom_color.blue())}
        return {"name": self.team2_color_combo.currentText(), "rgb": self.get_rgb_for_color(self.team2_color_combo.currentText())}
    
    def get_referee_color_data(self):
        if self.referee_color_combo.currentText() == "Custom...":
            return {"name": "Custom", "rgb": (self.referee_custom_color.red(), 
                                            self.referee_custom_color.green(), 
                                            self.referee_custom_color.blue())}
        return {"name": self.referee_color_combo.currentText(), "rgb": self.get_rgb_for_color(self.referee_color_combo.currentText())}
    
    def get_rgb_for_color(self, color_name):
        color_map = {
            "Red": (255, 0, 0),
            "Blue": (0, 0, 255),
            "Green": (0, 255, 0),
            "Yellow": (255, 255, 0),
            "Orange": (255, 165, 0),
            "Purple": (128, 0, 128),
            "Black": (0, 0, 0),
            "White": (255, 255, 255)
        }
        return color_map.get(color_name, (255, 255, 255))  # Default to white


class SoccerVisionApp(QWidget):    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soccer Vision App")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: white;")

        # Team configuration
        self.team1_name = "Team 1"
        self.team1_color = {"name": "Red", "rgb": (255, 0, 0)}
        self.team2_name = "Team 2"
        self.team2_color = {"name": "Blue", "rgb": (0, 0, 255)}
        self.referee_color = {"name": "Yellow", "rgb": (255, 255, 0)}

        # App state
        try:
            self.model = YOLO("data/best.pt")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a dummy model for testing
            self.model = None
        
        # Print model class names for debugging
        if self.model:
            print("Model classes:", self.model.names)
        
        # Define class mappings based on model output
        # 0: ball, 1: goalkeeper, 2: player, 3: referee
        self.class_mappings = {
            'ball': 0,
            'goalkeeper': 1,
            'player': 2,
            'referee': 3,
        }
        
        # Reverse mapping for display purposes
        self.class_names = {v: k for k, v in self.class_mappings.items()}
        
        if self.model:
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load logo, create a placeholder if not found
        try:
            self.logo = cv2.imread("data/logo.png", cv2.IMREAD_UNCHANGED)
            if self.logo is None:
                # Create a simple placeholder logo
                self.logo = np.ones((50, 100, 3), dtype=np.uint8) * 100
                cv2.putText(self.logo, "LOGO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            self.logo = None
            
        self.video_path = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # FPS calculation variables
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.fps_counter = 0
        
        self.team_counts = {self.team1_name: 0, self.team2_name: 0, "Referee": 0}
        self.current_frame_number = 0

        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title bar
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #3c4b64; color: white; border-radius: 5px;")
        title_frame.setMinimumHeight(40)
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("Soccer Vision App")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_layout.addWidget(title_label)
        main_layout.addWidget(title_frame)
        
        # Button bar
        button_frame = QFrame()
        button_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        button_layout = QHBoxLayout(button_frame)
        
        self.load_btn = QPushButton("Load Video")
        self.load_btn.setStyleSheet("background-color: #0275d8; color: white; padding: 8px 15px; border-radius: 3px; font-weight: bold;")
        self.load_btn.clicked.connect(self.load_video)
        
        self.config_btn = QPushButton("Configure Teams")
        self.config_btn.setStyleSheet("background-color: #6f42c1; color: white; padding: 8px 15px; border-radius: 3px; font-weight: bold;")
        self.config_btn.clicked.connect(self.configure_teams)
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setStyleSheet("background-color: #5cb85c; color: white; padding: 8px 15px; border-radius: 3px; font-weight: bold;")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Video")
        self.export_btn.setStyleSheet("background-color: #f0ad4e; color: white; padding: 8px 15px; border-radius: 3px; font-weight: bold;")
        self.export_btn.clicked.connect(self.export_video)
        self.export_btn.setEnabled(False)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.config_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addStretch()
        main_layout.addWidget(button_frame)
        
        # Video preview area
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: black; border-radius: 5px;")
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel("Load a video to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: none; color: white; font-size: 16px;")
        self.video_label.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_label)
        
        main_layout.addWidget(video_frame, 1)  # 1 = stretch factor to make it expand
        
        # Status bar
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px; padding: 5px;")
        status_frame.setMinimumHeight(60)
        status_layout = QVBoxLayout(status_frame)
        
        self.detection_label = QLabel(f"Detected: 🔴 {self.team1_name}: 0   🔵 {self.team2_name}: 0   🟡 Referee: 0")
        self.detection_label.setFont(QFont("Arial", 11, QFont.Bold))
        status_layout.addWidget(self.detection_label)
        
        self.frame_info_label = QLabel("Frame: 0   FPS: 0.0   Status: Ready")
        self.frame_info_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.frame_info_label)
        
        main_layout.addWidget(status_frame)
        
        self.setLayout(main_layout)
        
    def configure_teams(self):
        dialog = TeamConfigDialog(self)
        
        # Pre-populate with current values
        dialog.team1_name.setText(self.team1_name)
        dialog.team2_name.setText(self.team2_name)
        dialog.team1_color_combo.setCurrentText(self.team1_color["name"])
        dialog.team2_color_combo.setCurrentText(self.team2_color["name"])
        dialog.referee_color_combo.setCurrentText(self.referee_color["name"])
        
        if dialog.exec_() == QDialog.Accepted:
            # Update team configuration
            self.team1_name = dialog.team1_name.text()
            self.team2_name = dialog.team2_name.text()
            self.team1_color = dialog.get_team1_color_data()
            self.team2_color = dialog.get_team2_color_data()
            self.referee_color = dialog.get_referee_color_data()
            
            # Reset team counts with new names
            self.team_counts = {self.team1_name: 0, self.team2_name: 0, "Referee": 0, "Goalkeeper": 0, "Ball": 0}
            
            # Update status display
            self.update_status_display()
            
            print(f"Team configuration updated:")
            print(f"Team 1: {self.team1_name} ({self.team1_color})")
            print(f"Team 2: {self.team2_name} ({self.team2_color})")
            print(f"Referee: {self.referee_color}")
        
    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if self.video_path:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.video_label.setText("Error: Could not open video file")
                return
                
            self.current_frame_number = 0
            self.frame_count = 0
            self.start_time = time.time()
            self.fps = 0
            self.team_counts = {self.team1_name: 0, self.team2_name: 0, "Referee": 0, "Goalkeeper": 0, "Ball": 0}
            
            # Enable buttons
            self.start_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Display first frame
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.frame_info_label.setText(f"Frame: 1   FPS: 0.0   Status: Video Loaded")

    def start_detection(self):
        if not self.model:
            self.video_label.setText("Error: Model not loaded")
            return
            
        if self.cap:
            if not self.timer.isActive():
                self.timer.start(33)  # ~30 FPS
                self.start_btn.setText("Stop Detection")
                self.start_btn.setStyleSheet("background-color: #d9534f; color: white; padding: 8px 15px; border-radius: 3px; font-weight: bold;")
                self.frame_count = 0
                self.start_time = time.time()
            else:
                self.stop_detection()
    
    def stop_detection(self):
        self.timer.stop()
        self.start_btn.setText("Start Detection")
        self.start_btn.setStyleSheet("background-color: #5cb85c; color: white; padding: 8px 15px; border-radius: 3px; font-weight: bold;")
        
    def export_video(self):
        if not self.video_path or not self.model:
            return
            
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4)")
        if not output_path:
            return
            
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        # Process each frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_number += 1
            
            # Update progress in status
            progress = (frame_number / total_frames) * 100
            self.frame_info_label.setText(f"Exporting: {progress:.1f}% ({frame_number}/{total_frames})")
            QApplication.processEvents()  # Update UI
            
            # Process frame with model
            processed_frame = self.process_frame_for_export(frame)
            
            # Write frame
            out.write(processed_frame)
        
        # Release resources
        out.release()
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_info_label.setText(f"Export completed: {output_path}")
        
    def process_frame_for_export(self, frame):
        """Process a single frame for export"""
        results = self.model(frame)[0] # type: ignore
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            
            if conf < 0.5:  # Skip low confidence detections
                continue
            
            # Get class name from model's names dictionary
            class_name = self.class_names.get(cls, f"Unknown-{cls}")
            
            # Determine team for players and goalkeepers
            color = (255, 255, 255)  # Default color (white)
            
            if class_name in ["player", "goalkeeper"]:
                cropped = frame[y1:y2, x1:x2]
                detected_color = classify_team_color(cropped)
                
                # Map detected color to team
                team_name = "Unknown"
                if detected_color.lower() == self.team1_color["name"].lower():
                    team_name = self.team1_name
                    color = self.team1_color["rgb"][::-1]  # Convert RGB to BGR for OpenCV
                elif detected_color.lower() == self.team2_color["name"].lower():
                    team_name = self.team2_name
                    color = self.team2_color["rgb"][::-1]  # Convert RGB to BGR for OpenCV
                else:
                    # Try to match with known colors
                    color_mapping = {
                        "red": (0, 0, 255),
                        "blue": (255, 0, 0),
                        "green": (0, 255, 0),
                        "yellow": (0, 255, 255),
                        "purple": (128, 0, 128),
                        "black": (0, 0, 0),
                        "white": (255, 255, 255)
                    }
                    color = color_mapping.get(detected_color.lower(), (128, 128, 128))
                
                text = f"{class_name.capitalize()}"
                if team_name != "Unknown":
                    text += f" ({team_name})"
                else:
                    text += f" ({detected_color})"
            
            elif class_name == "referee":
                color = self.referee_color["rgb"][::-1]  # Convert RGB to BGR for OpenCV
                text = "Referee"
            
            elif class_name == "ball":
                color = (0, 0, 255)  # Red
                text = "Ball"
            
            else:
                text = class_name.capitalize()
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add logo
        frame = self.overlay_logo(frame)
        
        return frame
    
    def display_frame(self, frame):
        """Convert OpenCV frame to QPixmap and display it in the video_label"""
        if frame is None:
            return
            
        # Convert for Qt display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation) # type: ignore
        
        self.video_label.setPixmap(scaled_pixmap)

    def overlay_logo(self, frame):
        if self.logo is None:
            return frame
            
        logo_h, logo_w = self.logo.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        # Make sure logo fits in frame
        if logo_h >= frame_h - 20 or logo_w >= frame_w - 20:
            return frame
            
        overlay = frame.copy()
        
        # Check if logo has alpha channel
        if len(self.logo.shape) == 3 and self.logo.shape[2] == 4:  # Has alpha channel
            roi = overlay[10:10+logo_h, 10:10+logo_w]
            alpha_logo = self.logo[:, :, 3] / 255.0
            alpha_bg = 1.0 - alpha_logo
            
            for c in range(0, 3):
                roi[:, :, c] = (alpha_logo * self.logo[:, :, c] + alpha_bg * roi[:, :, c])
            
            overlay[10:10+logo_h, 10:10+logo_w] = roi
        else:  # No alpha channel - just overlay the logo
            overlay[10:10+logo_h, 10:10+logo_w] = self.logo
        
        return overlay

    def update_frame(self):
        if not self.cap or not self.model:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.frame_info_label.setText("Video ended - detection stopped")
            self.stop_detection()
            return

        # Update frame count and calculate FPS
        self.frame_count += 1
        self.fps_counter += 1
        current_time = time.time()
        time_diff = current_time - self.start_time
        
        if time_diff >= 1.0:  # Update FPS every second
            self.fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.start_time = current_time

        # Get current frame number from video
        self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Reset team counts for this frame
        self.team_counts = {self.team1_name: 0, self.team2_name: 0, "Referee": 0, "Goalkeeper": 0, "Ball": 0}
        
        # Process detections
        results = self.model(frame)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            
            if conf < 0.5:  # Skip low confidence detections
                continue
            
            # Get class name from model's names dictionary
            class_name = self.class_names.get(cls, f"Unknown-{cls}")
            
            # Determine team for players and goalkeepers
            color = (255, 255, 255)  # Default color (white)
            
            if class_name in ["player", "goalkeeper"]:
                cropped = frame[y1:y2, x1:x2]
                detected_color = classify_team_color(cropped)
                
                # Map detected color to team
                team_name = "Unknown"
                if detected_color.lower() == self.team1_color["name"].lower():
                    team_name = self.team1_name
                    color = self.team1_color["rgb"][::-1]  # Convert RGB to BGR for OpenCV
                elif detected_color.lower() == self.team2_color["name"].lower():
                    team_name = self.team2_name
                    color = self.team2_color["rgb"][::-1]  # Convert RGB to BGR for OpenCV
                else:
                    # Try to match with common colors
                    color_mapping = {
                        "red": (0, 0, 255),
                        "blue": (255, 0, 0),
                        "green": (0, 255, 0),
                        "yellow": (0, 255, 255),
                        "purple": (128, 0, 128),
                        "black": (0, 0, 0),
                        "white": (255, 255, 255)
                    }
                    color = color_mapping.get(detected_color.lower(), (128, 128, 128))
                
                # Update counts
                if class_name == "player":
                    if team_name in [self.team1_name, self.team2_name]:
                        self.team_counts[team_name] += 1
                elif class_name == "goalkeeper":
                    self.team_counts["Goalkeeper"] += 1
                
                # Text to display
                text = f"{class_name.capitalize()}"
                if team_name != "Unknown":
                    text += f" ({team_name})"
                else:
                    text += f" ({detected_color})"
            
            elif class_name == "referee":
                color = self.referee_color["rgb"][::-1]  # Convert RGB to BGR for OpenCV
                text = "Referee"
                self.team_counts["Referee"] += 1
            
            elif class_name == "ball":
                color = (0, 0, 255)  # Red
                text = "Ball"
                self.team_counts["Ball"] += 1
            
            else:
                text = class_name.capitalize()
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw text background for better visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Update status displays
        self.update_status_display()
        self.frame_info_label.setText(f"Frame: {self.current_frame_number}   FPS: {self.fps:.1f}   Status: Detecting")

        # Add logo overlay
        frame = self.overlay_logo(frame)

        # Display the frame
        self.display_frame(frame)
        
    def update_status_display(self):
        """Update the detection status display"""
        status_text = f"Detected: 🔴 {self.team1_name}: {self.team_counts[self.team1_name]}  "
        status_text += f"🔵 {self.team2_name}: {self.team_counts[self.team2_name]}  "
        status_text += f"🧤 Goalkeeper: {self.team_counts['Goalkeeper']}  "
        status_text += f"🟡 Referee: {self.team_counts['Referee']}  "
        status_text += f"⚽ Ball: {self.team_counts['Ball']}"
        
        self.detection_label.setText(status_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SoccerVisionApp()
    win.show()
    sys.exit(app.exec_())