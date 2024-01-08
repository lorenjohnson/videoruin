import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import argparse

def convert_frame(frame, target_width, target_height):
    # Your image processing logic here
    # Example: convert to grayscale and resize
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame = cv2.resize(gray_frame, (target_width, target_height))

    return small_frame

def apply_color_dithering(frame):
    # Convert the frame to a Pillow Image
    pil_frame = Image.fromarray(frame)

    # Apply color dithering (Floyd-Steinberg dithering in this case)
    dithered_frame = pil_frame.convert('P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG)

    # Convert the Pillow Image back to a NumPy array
    dithered_frame_np = np.array(dithered_frame)

    return dithered_frame_np

def crop_middle_band(frame, target_width, target_height):
    # Calculate the middle band based on the aspect ratio of the target resolution
    aspect_ratio = target_width / target_height
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    middle_band_height = int(frame_width / aspect_ratio)
    middle_band_start = (frame_height - middle_band_height) // 2
    middle_band_end = middle_band_start + middle_band_height

    return frame[middle_band_start:middle_band_end, :]

def process_video(input_path, output_path, target_width, target_height):
    video_clip = VideoFileClip(input_path)

    # Create VideoWriter objects with moviepy
    output_video_path = os.path.join(output_path, f'output_{os.path.basename(input_path)}')
    output_top_band_path = os.path.join(output_path, f'top_band_{os.path.basename(input_path)}')
    output_bottom_band_path = os.path.join(output_path, f'bottom_band_{os.path.basename(input_path)}')

    out = cv2.VideoWriter(output_video_path, fourcc, video_clip.fps, (target_width, target_height))
    out_top_band = cv2.VideoWriter(output_top_band_path, fourcc, video_clip.fps, (target_width, target_height))
    out_bottom_band = cv2.VideoWriter(output_bottom_band_path, fourcc, video_clip.fps, (target_width, target_height))

    frame_count = int(video_clip.fps * video_clip.duration)

    for i in range(frame_count):
        frame = video_clip.get_frame(i / video_clip.fps)

        processed_frame = convert_frame(frame, target_width, target_height)
        dithered_frame = apply_color_dithering(processed_frame)

        # Crop the middle band based on the aspect ratio of the target resolution
        middle_band_frame = crop_middle_band(dithered_frame, target_width, target_height)

        # Save the middle band to the main video
        out.write(cv2.cvtColor(middle_band_frame, cv2.COLOR_GRAY2BGR))

        # Save the top and bottom bands to separate videos
        top_band_frame = dithered_frame[:middle_band_frame.shape[0], :]
        bottom_band_frame = dithered_frame[-middle_band_frame.shape[0]:, :]
        out_top_band.write(cv2.cvtColor(top_band_frame, cv2.COLOR_GRAY2BGR))
        out_bottom_band.write(cv2.cvtColor(bottom_band_frame, cv2.COLOR_GRAY2BGR))

    video_clip.reader.close()
    out.release()
    out_top_band.release()
    out_bottom_band.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and generate output videos.")
    parser.add_argument("input_directory", help="Path to the directory containing input videos.")
    parser.add_argument("output_directory", help="Path to the directory for saving output videos.")
    parser.add_argument("--width", type=int, required=True, help="Target width for output videos.")
    parser.add_argument("--height", type=int, required=True, help="Target height for output videos.")
    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    target_width = args.width
    target_height = args.height

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Loop through each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            video_path = os.path.join(input_directory, filename)
            process_video(video_path, output_directory, target_width, target_height)

    print("Processing complete.")
