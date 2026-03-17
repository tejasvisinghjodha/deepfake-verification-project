import cv2          # OpenCV library for video processing
import os           # For handling file paths and folders

def extract_frames(video_path, interval_seconds=1):
    """
    Opens a video file and extracts one frame every `interval_seconds` seconds.

    Args:
        video_path       : Path to the video file (e.g., "sample.mp4")
        interval_seconds : How often to capture a frame (default = every 1 second)

    Returns:
        frames : A list of images (each image is a NumPy array)
    """

    # Step A: Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Step B: Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return []

    # Step C: Get the video's frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # Step D: Calculate how many frames to skip between each capture
    # e.g., if FPS=30 and interval=1 second, capture every 30th frame
    frame_interval = int(fps * interval_seconds)

    frames = []         # This list will store all the captured frames
    frame_count = 0     # Counter to track which frame we're on

    # Step E: Loop through the video frame by frame
    while True:
        success, frame = cap.read()   # Read the next frame

        # If no more frames, stop the loop
        if not success:
            break

        # Step F: Only save the frame if it falls on our interval
        if frame_count % frame_interval == 0:
            frames.append(frame)      # Add the frame (image) to our list
            print(f"Captured frame at position: {frame_count}")

        frame_count += 1  # Move to the next frame

    # Step G: Release the video file from memory
    cap.release()

    print(f"Total frames extracted: {len(frames)}")
    return frames


def get_video_metadata(video_path):
    """
    Extracts basic metadata from a video file.

    Returns:
        A dictionary with fps, total frames, width, height, duration
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {}

    fps            = cap.get(cv2.CAP_PROP_FPS)
    total_frames   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width          = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height         = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration_secs  = total_frames / fps if fps > 0 else 0

    cap.release()

    metadata = {
        "fps"           : fps,
        "total_frames"  : total_frames,
        "width"         : int(width),
        "height"        : int(height),
        "duration_secs" : round(duration_secs, 2)
    }

    return metadata


def save_frames(frames, output_folder="extracted_frames"):
    """
    Saves extracted frames as image files on disk.
    Useful for visual testing — you can open them and see what was captured.

    Args:
        frames        : List of frames returned by extract_frames()
        output_folder : Folder name where images will be saved
    """
    # Create the folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        filename = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(filename, frame)   # Save the image to disk
        print(f"Saved: {filename}")

    print(f"All frames saved to '{output_folder}/' folder.")
