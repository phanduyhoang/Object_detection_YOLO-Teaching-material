from ultralytics import YOLO
import cv2

# Load YOLOv5 pre-trained model
model = YOLO('yolov5su.pt')  # Load the YOLOv5 model

def process_video(video_path, output_path, conf=0.25):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break when video ends
        
        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv5 prediction (detect all classes)
        results = model.predict(source=rgb_frame, imgsz=640, conf=conf)
        
        # Annotate the frame
        annotated_frame = results[0].plot()

        # Convert annotated frame back to BGR for OpenCV
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(annotated_frame_bgr)

        # Display the frame in a window
        cv2.imshow('YOLOv5 Detection', annotated_frame_bgr)

        # Wait for 'q' key to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process the video and detect everything
process_video('input_video.mp4', 'output_video.mp4', conf=0.25)
