from moviepy.editor import VideoFileClip, AudioFileClip
import cv2
from gradio_client import Client
import os

def extract_firstframe(video_in):
    """Extract the first frame from video for caption generation"""
    vidcap = cv2.VideoCapture(video_in)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("first_frame.jpg", image)
        return "first_frame.jpg"
    return None

def get_caption(image_path):
    """Get caption for the video frame using Moondream model"""
    client = Client("fffiloni/moondream1")
    result = client.predict(
        image=image_path,
        question="Describe precisely the image in one sentence.",
        api_name="/predict"
    )
    return result

def generate_audio(caption, model="Tango"):
    """Generate audio based on caption using selected model"""
    if model == "Tango":
        client = Client("fffiloni/tango")
        result = client.predict(
            caption,
            100,  # steps
            4,    # guidance scale
            api_name="/predict"
        )
        return result
    # Add other model options as needed

def blend_audio_video(video_path, audio_path):
    """Combine generated audio with original video"""
    audio_clip = AudioFileClip(audio_path)
    video_clip = VideoFileClip(video_path)
    
    # Match durations
    if video_clip.duration < audio_clip.duration:
        audio_clip = audio_clip.subclip(0, video_clip.duration)
    elif video_clip.duration > audio_clip.duration:
        video_clip = video_clip.subclip(0, audio_clip.duration)
    
    final_clip = video_clip.set_audio(audio_clip)
    output_path = 'output_video_with_sound.mp4'
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    return output_path

def process_video(video_path, model="Tango"):
    """Main function to process video and generate matching audio"""
    # Extract first frame
    frame_path = extract_firstframe(video_path)
    if not frame_path:
        raise Exception("Failed to extract frame from video")
    
    # Generate caption
    caption = get_caption(frame_path)
    print(f"Generated caption: {caption}")
    
    # Generate audio
    audio_path = generate_audio(caption, model)
    print(f"Generated audio saved to: {audio_path}")
    
    # Combine audio and video
    output_path = blend_audio_video(video_path, audio_path)
    print(f"Final video saved to: {output_path}")
    
    return output_path
