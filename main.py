import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
from typing import List
import asyncio
import tempfile
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import time

app = FastAPI()
#the model is cahngebale
#just change the plug in layers adn pararmeters in the processing.
model = load_model('EmotionK.h5')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
last_valence = 0  # Initial values; they'll be updated once a face is detected
last_arousal = 0
def safe_delete(file_path, max_attempts=5, delay_between_attempts=0.1):
    for _ in range(max_attempts):
        try:
            os.remove(file_path)
            break  # If successful, exit the loop
        except PermissionError:
            time.sleep(delay_between_attempts)  # Wait and then try again
    else:  # This part will run if the loop completed without a break
        print(f"Failed to delete {file_path} after {max_attempts} attempts.")
def preprocess_video(video: bytes) -> bytes:
    # Create a temporary file to store the input video
    input_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    input_temp_file.write(video)
    input_temp_file.close()

    # Create a VideoCapture object from the temporary file
    video_capture = cv2.VideoCapture(input_temp_file.name)

    
    # process frame to generate the valence arousal graph in this funtion
    frame_list = []

    def process_frame(frame):
        global last_valence, last_arousal # use global to keep track across frames

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            # Use the last known valence and arousal if no face is detected
            valence = last_valence
            arousal = last_arousal
            print("No face detected. Using the latest valence and arousal values.")
        else:
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]

            # Resize the face to 224x224 pixels
            resized_face = cv2.resize(face, (224, 224))
            normalized_face = resized_face / 255.0
            face_array = normalized_face.reshape((1, 224, 224, 3))

            # Pass the face through the model and get the output
            output = model.predict(face_array)
            valence = np.clip(output[0][0]*5, -10, 10)
            arousal = np.clip(output[0][1]*3, -10, 10)
            
            # Update the last known valence and arousal
            last_valence = valence
            last_arousal = arousal

    

        print("Valence:", valence)
        print("Arousal:", arousal)
        img = cv2.imread('background_img.png')
        
        

        # Define the image center as the origin of the graph
        origin = (img.shape[1] // 2, img.shape[0] // 2)
        # Create a plot
        fig, ax = plt.subplots()
        plt.imshow(img, extent=[-10, 10, -10, 10])
        ax.scatter(valence, arousal, s=50, c='red')
        ax.set_xlabel('Valence Score')
        ax.set_ylabel('Arousal Score')
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        x_ticks = np.linspace(x_min, x_max, 10)
        y_ticks = np.linspace(y_min, y_max, 10)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(x_ticks)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.axhline(y=origin[1], color='gray')
        ax.axvline(x=origin[0], color='gray')

        # Show the plot
        
        fig.savefig(f"frame_{num_frames}.png")
        plt.close(fig)
        # Append the plot to the frame list

        frame_list.append(np.array(fig.canvas.renderer.buffer_rgba()))

        # Clear the plot
        ax.cla()
        plt.close(fig)
        return frame_list
    # Initialize a list to store processed frames
    processed_frames = []
    num_frames=-1
    while True:
        num_frames=num_frames+1
        ret, frame = video_capture.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame)
        processed_frames.append(processed_frame)
        
    # Release the VideoCapture object
    video_capture.release()

    # Delete the temporary input file
    os.unlink(input_temp_file.name)

    # Create a temporary file to store the output video
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_temp_file.close()
    fps = 10
    # Write the processed frames into the temporary output file as a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    image=cv2.imread('frame_0.png')
    #plot_frame=Image.fromarray(np.uint8(processed_frame[0]))
    #output_video = cv2.VideoWriter(output_temp_file.name, -1, fps, (640, 480), isColor=False)
    #output_video =cv2.VideoWriter(output_temp_file.name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))
    output_video = cv2.VideoWriter(output_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    # Define the video codec and frame rate


    # Add each frame to the video
    for i in range(num_frames):
        frame_path = f"frame_{i}.png"
        frame = cv2.imread(frame_path)
        output_video.write(frame)
        if os.path.exists(frame_path):  # Only try to remove the file if it exists
            safe_delete(frame_path)
        
    output_video.release()


    # Read the bytes of the processed video from the temporary output file
    with open(output_temp_file.name, 'rb') as f:
        output_video_bytes = f.read()

    # Delete the temporary output file
    os.unlink(output_temp_file.name)

    # Return the bytes of the processed video
    return output_video_bytes

@app.post("/process_video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    video = await file.read()
    processed_video = preprocess_video(video)
    response = StreamingResponse(BytesIO(processed_video), media_type="video/mp4")
    response.headers["Content-Disposition"] = f"attachment; filename=processed_{file.filename}"
    
    return response
