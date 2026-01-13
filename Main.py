import cv2
import math
import time
import sqlite3
from datetime import datetime
from ultralytics import YOLO

# --- DATABASE SETUP ---
def init_db():
    """Creates the database and table if they don't exist."""
    conn = sqlite3.connect('traffic_data.db')
    cursor = conn.cursor()
    # Create table to store Vehicle ID, Class (Car/Truck), Speed, and Time
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            vehicle_type TEXT,
            speed INTEGER,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()
    print("Database Initialized Successfully.")

def log_vehicle(v_id, v_type, v_speed):
    """Saves a single vehicle's data to the SQL database."""
    conn = sqlite3.connect('traffic_data.db')
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('INSERT INTO vehicle_logs (vehicle_id, vehicle_type, speed, timestamp) VALUES (?, ?, ?, ?)',
                   (v_id, v_type, v_speed, current_time))
    
    conn.commit()
    conn.close()
    print(f"Logged Vehicle ID {v_id} ({v_type}) at {v_speed} km/h")

# Initialize DB immediately
init_db()

# --- CONFIGURATION ---
COUNT_LINE_POSITION = 350 
OFFSET = 6 

# Load Model
print("Loading YOLOv8 Model...")
model = YOLO('yolov8n.pt') 

# Video Source
cap = cv2.VideoCapture('video.mp4') 
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

# Tracker Storage
vehicle_tracker = {}
vehicle_speeds = {}
counter_set = set() # Keeps track of IDs we have already saved

print("Starting System...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Run Tracking
    results = model.track(frame, persist=True, stream=True)

    # Draw Line
    cv2.line(frame, (2, COUNT_LINE_POSITION), (1018, COUNT_LINE_POSITION), (0, 0, 255), 2)
    cv2.putText(frame, "Detection Line", (10, COUNT_LINE_POSITION - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for r in results:
        boxes = r.boxes
        
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int) # Get class IDs
            
            for box_id, box, cls in zip(ids, coords, clss):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Get Class Name (Car, Truck, etc.)
                class_name = model.names[cls]

                # Speed Logic
                speed_est = 0
                if box_id in vehicle_tracker:
                    prev_x, prev_y, prev_time = vehicle_tracker[box_id]
                    curr_time = time.time()
                    distance_pixels = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    time_diff = curr_time - prev_time
                    if time_diff > 0:
                        speed_est = int((distance_pixels / time_diff) * 0.1)
                        vehicle_speeds[box_id] = speed_est

                vehicle_tracker[box_id] = (cx, cy, time.time())

                # --- DB LOGGING LOGIC ---
                # Only log if it crosses the line AND hasn't been counted yet
                if (COUNT_LINE_POSITION - OFFSET) < cy < (COUNT_LINE_POSITION + OFFSET):
                    if box_id not in counter_set:
                        counter_set.add(box_id)
                        
                        # SAVE TO DATABASE HERE
                        log_vehicle(int(box_id), class_name, speed_est)
                        
                        # Visual feedback
                        cv2.line(frame, (2, COUNT_LINE_POSITION), (1018, COUNT_LINE_POSITION), (0, 255, 0), 2)

                # Visualization
                speed_text = f"{vehicle_speeds.get(box_id, 0)} km/h"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"ID:{box_id} {class_name}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Display Count
    cv2.putText(frame, f'Logged: {len(counter_set)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Major Project - Database Integrated", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()