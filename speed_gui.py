import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
from PIL import Image, ImageTk

class SpeedDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Speed Detection")
        
        # Create main container
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for video
        self.video_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.video_frame)
        
        # Right panel for speed display
        self.speed_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.speed_frame)
        
        # Video canvas
        self.canvas = tk.Canvas(self.video_frame, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Speed display
        self.speed_label = ttk.Label(
            self.speed_frame, 
            text="Fastest Approaching Vehicle",
            font=('Arial', 20)
        )
        self.speed_label.pack(pady=20)
        
        self.speed_value = ttk.Label(
            self.speed_frame,
            text="0 km/h",
            font=('Arial', 40, 'bold')
        )
        self.speed_value.pack(pady=20)
        
        # Vehicle type label
        self.vehicle_type_label = ttk.Label(
            self.speed_frame,
            text="Vehicle Type",
            font=('Arial', 20)
        )
        self.vehicle_type_label.pack(pady=10)

        # Vehicle type value (will be updated dynamically)
        self.vehicle_type_value = ttk.Label(
            self.speed_frame,
            text="Unknown",
            font=('Arial', 30, 'bold')
        )
        self.vehicle_type_value.pack(pady=20)

        
        # Initialize video processing components
        self.setup_video_processor()
        
        # Start video processing
        self.process_video()
        
    def setup_video_processor(self):
        # Configuration variables
        self.SOURCE_VIDEO_PATH = "C:/Users/LENOVO/Desktop/FYP/Dynamic Vehicle Speed Detection/vehicles.mp4"  # Update with your video path
        self.CONFIDENCE_THRESHOLD = 0.3
        self.IOU_THRESHOLD = 0.7
        
        # Perspective transform configuration
        self.SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
        self.TARGET_WIDTH = 25
        self.TARGET_HEIGHT = 250
        self.TARGET = np.array([
            [0, 0],
            [self.TARGET_WIDTH - 1, 0],
            [self.TARGET_WIDTH - 1, self.TARGET_HEIGHT - 1],
            [0, self.TARGET_HEIGHT - 1],
        ])
        
        # Initialize components
        self.video_info = sv.VideoInfo.from_video_path(self.SOURCE_VIDEO_PATH)
        self.model = YOLO("yolov8n.pt")
        self.byte_track = sv.ByteTrack(
            frame_rate=self.video_info.fps,
            track_activation_threshold=self.CONFIDENCE_THRESHOLD
        )
        
        # Initialize annotators
        thickness = sv.calculate_optimal_line_thickness(self.video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(self.video_info.resolution_wh)
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=self.video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )
        
        # Initialize other components
        self.polygon_zone = sv.PolygonZone(polygon=self.SOURCE)
        self.view_transformer = ViewTransformer(source=self.SOURCE, target=self.TARGET)
        self.coordinates = defaultdict(lambda: deque(maxlen=self.video_info.fps))
        
        # Open video capture
        self.cap = cv2.VideoCapture(self.SOURCE_VIDEO_PATH)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def is_vehicle_approaching(self, coordinates):
        """Check if vehicle is moving towards the screen (y increasing)"""
        if len(coordinates) >= 2:
            start_y = coordinates[0]
            end_y = coordinates[-1]
            return end_y > start_y
        return False
        
    def process_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            progress = (self.current_frame / self.total_frames) * 100
            print(f"Processing frame {self.current_frame}/{self.total_frames} ({progress:.1f}%)")
            
            # Process frame
            result = self.model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > self.CONFIDENCE_THRESHOLD]
            detections = detections[self.polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=self.IOU_THRESHOLD)
            detections = self.byte_track.update_with_detections(detections=detections)
            
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = self.view_transformer.transform_points(points=points).astype(int)
            
            # Initialize tracking variables
            max_speed = 0
            fastest_vehicle_type = "Unknown"
            labels = []

            for tracker_id, [_, y], class_id in zip(detections.tracker_id, points, detections.class_id):
                self.coordinates[tracker_id].append(y)
                
                if len(self.coordinates[tracker_id]) >= self.video_info.fps / 2:
                    if self.is_vehicle_approaching(self.coordinates[tracker_id]):
                        coordinate_start = self.coordinates[tracker_id][-1]
                        coordinate_end = self.coordinates[tracker_id][0]
                        distance = abs(coordinate_start - coordinate_end)
                        time = len(self.coordinates[tracker_id]) / self.video_info.fps
                        speed = distance / time * 3.6
                        
                        if speed > max_speed:
                            max_speed = speed
                            fastest_vehicle_type = self.model.names.get(class_id, "Unknown")  # Safe lookup for vehicle type
                        
                        labels.append(f"#{tracker_id} {int(speed)} km/h")
                    else:
                        labels.append(f"#{tracker_id}")
                else:
                    labels.append(f"#{tracker_id}")

            # Update GUI with fastest vehicle speed and type
            self.speed_value.config(text=f"{int(max_speed)} km/h")
            self.vehicle_type_value.config(text=fastest_vehicle_type.capitalize())
            
            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize frame to fit canvas
            frame_pil = frame_pil.resize((800, 600), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(frame_pil)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # Schedule next frame
            self.root.after(30, self.process_video)
        else:
            # Clean up and close when video ends
            self.cap.release()
            print("\nVideo processing completed!")
            self.root.quit()
            self.root.destroy()

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedDetectionGUI(root)
    root.mainloop()
