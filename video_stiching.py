from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def stitch_videos(video_paths, output_path='output.mp4'):
    """
    Stitch multiple video files together into a single video file.
    
    Args:
        video_paths (list): List of paths to video files to be stitched
        output_path (str): Path where the output video will be saved
        
    Returns:
        str: Path to the stitched video file
    """
    try:
        # Load all video clips
        clips = [VideoFileClip(path) for path in video_paths]
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips, method='compose')
        
        # Write the result to a file
        final_clip.write_videofile(output_path)
        
        # Close all clips to free up resources
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    video_files = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
        "path/to/video3.mp4"
    ]
    
    output = stitch_videos(video_files, "final_video.mp4")
    if output:
        print(f"Videos successfully stitched. Output saved to: {output}")
