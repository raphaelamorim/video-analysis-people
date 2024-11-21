import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

class PersonDetector:
    def __init__(self):
        self.model = YOLO('yolo11n.pt')
        self.colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (255, 0, 255),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255)
        }
        
    def get_model_id(self, name):
        for key, value in self.model.names.items():
            if value == name:
                return key
        return None

    def process_frame(self, frame):
        if frame is None:
            return None
            
        # Convert PIL Image to numpy array if needed
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
            
        # Ensure frame is in BGR format for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        # Run detection
        results = self.model(frame, classes=[self.get_model_id('person')])  # class 0 is person
        
        color = tuple(np.flip(self.colors['green']).tolist())
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add confidence score
                conf = float(box.conf[0])
                cv2.putText(frame, f'Person: {conf:.2f}', 
                          (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, color, 2)
        
        # Convert back to RGB for Gradio
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def process_video(self, video_path):
        """Process video file and return processed video file path"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output file in the same directory as input
        output_path = os.path.join(
            os.path.dirname(video_path),
            'processed_' + os.path.basename(video_path)
        )
        
        # Initialize video writer
        output = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Convert back to BGR for video writing
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            output.write(processed_frame)
            
        # Clean up
        cap.release()
        output.release()
        
        return output_path

def create_gradio_interface():
    detector = PersonDetector()
    
    def process_video_file(video_file):
        """Handler for video file input"""
        if video_file is None:
            return None
            
        # Process the video
        processed_video_path = detector.process_video(video_file)
        
        # Return the processed video file path
        return processed_video_path
    
    def process_webcam(frame):
        """Process webcam frame"""
        if frame is None:
            return None
        return detector.process_frame(frame)
    
    # Create interface
    with gr.Blocks(title="Person Detection App") as interface:
        gr.Markdown("# Person Detection Application")
        gr.Markdown("Upload a video file or use your webcam to detect people in real-time.")
        
        with gr.Tabs():
            with gr.TabItem("Video File"):
                with gr.Row():
                    video_input = gr.Video(label="Input Video")
                    video_output = gr.Video(label="Processed Video")
                process_btn = gr.Button("Process Video")
                process_btn.click(
                    fn=process_video_file,
                    inputs=video_input,
                    outputs=video_output
                )
            
            with gr.TabItem("Webcam"):
                with gr.Row():
                    webcam = gr.Image(sources="webcam", type="numpy", label="Webcam Feed")
                    output_video = gr.Image(label="Processed Feed", streaming=True)
                
                # Stream processing
                webcam.stream(
                    process_webcam,
                    webcam,
                    output_video
                )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    app = create_gradio_interface()
    app.launch(share=True)