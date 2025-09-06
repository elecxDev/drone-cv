#!/usr/bin/env python3
"""
Live Vehicle Detection with Real-time Display

This script provides live video feed with vehicle detection and proper unique vehicle counting.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
video_path = sys.argv[1]
def run_live_detection():
    """Run vehicle detection with live preview and unique vehicle tracking"""
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # nano version for speed
    print("YOLO model loaded successfully!")
    
    # Define vehicle-related classes from COCO dataset (removed trains - not relevant for aerial footage)
    # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 8: boat
    vehicle_classes = [0, 1, 2, 3, 5, 7, 8]
    class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
        5: 'bus', 7: 'truck', 8: 'boat'
    }
    
    # Color mapping for different vehicle types
    class_colors = {
        0: (255, 100, 100),  # person - light blue
        1: (0, 255, 255),    # bicycle - yellow
        2: (0, 255, 0),      # car - green
        3: (255, 0, 255),    # motorcycle - magenta
        5: (0, 0, 255),      # bus - red
        7: (255, 255, 0),    # truck - cyan
        8: (128, 0, 128)     # boat - purple
    }
    
    # Video writer for output (optional)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('live_detected_output.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    unique_vehicles = set()  # Track unique vehicles across frames
    detection_summary = {}
    vehicle_persistence = {}  # Track vehicle locations over time to handle temporary detection gaps
    
    # Calculate display size (scale down for better viewing on screen)
    display_scale = 0.5  # Adjust this value based on your screen size
    display_width = int(width * display_scale)
    display_height = int(height * display_scale)
    
    print(f"\nProcessing {total_frames} frames...")
    print("=" * 60)
    print("🎮 CONTROLS:")
    print("   'q' - Quit")
    print("   'p' - Pause/Resume")
    print("   's' - Save screenshot")
    print("   'r' - Reset vehicle count")
    print("   SPACE - Step frame (when paused)")
    print("=" * 60)
    print("Live preview window will open...")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached!")
                break
            
            frame_count += 1
            
            # Run detection with lower confidence threshold
            results = model(frame, conf=0.25, verbose=False)
            
            frame_detections = 0
            current_frame_vehicles = []
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's a vehicle class
                        if class_id in vehicle_classes:
                            frame_detections += 1
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Improved unique identifier for this detection
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            box_area = (x2 - x1) * (y2 - y1)
                            
                            # Much larger spatial binning to handle vehicle movement and detection gaps
                            # Use larger bins based on video resolution
                            bin_size_x = max(100, width // 20)  # Divide width into ~20 zones
                            bin_size_y = max(100, height // 20)  # Divide height into ~20 zones
                            
                            grid_x = center_x // bin_size_x
                            grid_y = center_y // bin_size_y
                            
                            # Create vehicle ID based on class and larger grid position
                            base_vehicle_id = f"{class_id}_{grid_x}_{grid_y}"
                            
                            # Check nearby grid cells for existing vehicles of same type
                            # This handles vehicles that move between grid cells
                            found_existing = False
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nearby_id = f"{class_id}_{grid_x + dx}_{grid_y + dy}"
                                    if nearby_id in vehicle_persistence:
                                        # Use existing vehicle ID if found nearby
                                        base_vehicle_id = nearby_id
                                        found_existing = True
                                        break
                                if found_existing:
                                    break
                            
                            # Update vehicle persistence (track last seen frame)
                            vehicle_persistence[base_vehicle_id] = frame_count
                            current_frame_vehicles.append(base_vehicle_id)
                            
                            # Update detection summary
                            class_name = class_names[class_id]
                            if class_name not in detection_summary:
                                detection_summary[class_name] = set()
                            detection_summary[class_name].add(base_vehicle_id)
                            
                            # Get color for this vehicle type
                            color = class_colors.get(class_id, (128, 128, 128))
                            
                            # Draw bounding box with thicker lines
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                            
                            # Add label with confidence
                            label = f'{class_name}: {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                            
                            # Draw label background
                            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                                        (x1 + label_size[0] + 10, y1), color, -1)
                            
                            # Draw label text
                            cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Clean up old vehicle IDs that haven't been seen recently (every 30 frames)
            if frame_count % 30 == 0:
                frames_to_keep = 150  # Keep vehicles seen in last 150 frames (about 6 seconds at 25fps)
                old_vehicles = []
                for vehicle_id, last_seen in vehicle_persistence.items():
                    if frame_count - last_seen > frames_to_keep:
                        old_vehicles.append(vehicle_id)
                
                # Remove old vehicles from tracking
                for old_id in old_vehicles:
                    del vehicle_persistence[old_id]
                    unique_vehicles.discard(old_id)
                    
                    # Remove from detection summary too
                    for class_name in list(detection_summary.keys()):
                        detection_summary[class_name].discard(old_id)
                        if len(detection_summary[class_name]) == 0:
                            del detection_summary[class_name]
            
            # Update unique vehicles set
            unique_vehicles.update(current_frame_vehicles)
            
            # Calculate unique vehicle counts
            total_unique_vehicles = len(unique_vehicles)
            unique_by_class = {class_name: len(vehicles) for class_name, vehicles in detection_summary.items()}
            
            # Create separate info panel (instead of overlaying on video)
            info_panel_width = 400
            info_panel_height = max(300, 200 + len(unique_by_class) * 30)
            info_panel = np.zeros((info_panel_height, info_panel_width, 3), dtype=np.uint8)
            
            # Add border to info panel
            cv2.rectangle(info_panel, (5, 5), (info_panel_width-5, info_panel_height-5), (255, 255, 255), 2)
            
            # Add title
            cv2.putText(info_panel, 'DETECTION INFO', (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add frame and detection info
            y_pos = 60
            cv2.putText(info_panel, f'Frame: {frame_count}/{total_frames}', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_pos += 25
            cv2.putText(info_panel, f'This frame: {frame_detections}', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            y_pos += 35
            cv2.putText(info_panel, f'UNIQUE VEHICLES: {total_unique_vehicles}', 
                       (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add class breakdown
            if unique_by_class:
                y_pos += 35
                cv2.putText(info_panel, 'Vehicle Types:', 
                           (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 25
                
                for class_name, count in unique_by_class.items():
                    color = class_colors.get([k for k, v in class_names.items() if v == class_name][0], (255, 255, 255))
                    cv2.putText(info_panel, f'  {class_name}: {count}', 
                               (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 25
            
            # Add progress bar
            progress = frame_count / total_frames
            bar_width = info_panel_width - 30
            bar_height = 15
            bar_x, bar_y = 15, info_panel_height - 40
            
            # Background
            cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Progress
            cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            # Border
            cv2.rectangle(info_panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
            
            # Progress text
            cv2.putText(info_panel, f'{progress*100:.1f}%', 
                       (bar_x + 5, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Create combined display: video + info panel side by side
            # Resize frame for display first
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # Resize info panel to match video height
            info_panel_resized = cv2.resize(info_panel, (info_panel_width, display_height))
            
            # Combine horizontally
            combined_display = np.hstack((display_frame, info_panel_resized))
            
            # Write original frame to output video (without info panel)
            out.write(frame)
        
        # Show live preview with info panel on the side
        cv2.imshow('🚁 Drone Vehicle Detection - LIVE FEED', combined_display)
        
        # Handle keyboard input
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"Video {status}")
        elif key == ord('s'):
            screenshot_name = f'screenshot_frame_{frame_count}.jpg'
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved: {screenshot_name}")
        elif key == ord('r'):
            unique_vehicles.clear()
            detection_summary.clear()
            print("Vehicle count reset!")
        elif key == ord(' ') and paused:
            # Step one frame when paused
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                print(f"Stepped to frame {frame_count}")
        
        # Print progress every 50 frames (only when not paused)
        if not paused and frame_count % 50 == 0:
            progress_pct = (frame_count / total_frames) * 100
            print(f"Progress: {progress_pct:.1f}% | Frame {frame_count}/{total_frames} | Unique vehicles: {total_unique_vehicles}")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print final results
    print("=" * 60)
    print(f"🎉 DETECTION COMPLETE!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total UNIQUE vehicles detected: {total_unique_vehicles}")
    print(f"Output video saved as: live_detected_output.mp4")
    
    if detection_summary:
        print(f"\n📊 Final unique vehicle breakdown:")
        for class_name, vehicles in detection_summary.items():
            print(f"   {class_name}: {len(vehicles)} unique vehicles")
    else:
        print("\n❓ No vehicles detected in the video.")
        print("Possible reasons:")
        print("   • No vehicles present in aerial footage")
        print("   • Vehicles too small/distant to detect")
        print("   • Video quality/lighting issues")
        print("   • Need to adjust detection parameters")

if __name__ == "__main__":
    print("🚁 Live Drone Vehicle Detection")
    print("=" * 60)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Make sure the video file is in the current directory.")
        sys.exit(1)
    
    try:
        run_live_detection()
    except KeyboardInterrupt:
        print("\n\n⏹️  Detection stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
