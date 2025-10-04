import os
import glob
import json
import requests
import fal_client
import openai
import subprocess
from dotenv import load_dotenv
# from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

def load_fal_api_key():
    """
    Load the FAL_KEY from the .env file and set it as environment variable
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the FAL API key from environment variables
    fal_api_key = os.getenv('FAL_KEY')
    
    if not fal_api_key:
        raise ValueError("FAL_KEY not found in environment variables. Please check your .env file.")
    
    # Set the FAL_KEY environment variable for fal_client
    os.environ['FAL_KEY'] = fal_api_key
    
    return fal_api_key

def load_openai_api_key():
    """
    Load the OPENAI_API_KEY from the .env file
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the OpenAI API key from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    
    return openai_api_key

def generate_video_script_to_tts_audio(story_prompt):
    """
    Generate a video script using OpenAI, divided into short verses for TTS.
    Each verse focuses on a single scene for image generation.
    
    Args:
        story_prompt (str): The main story prompt/theme for the video
    
    Returns:
        list: List of script verses, each describing a single scene
    """
    try:
        # Load OpenAI API key
        api_key = load_openai_api_key()
        
        # Set the API key for OpenAI
        openai.api_key = api_key
        
        print(f"Generating video script for: {story_prompt}")
        
        # Create a detailed prompt for script generation
        system_prompt = """You are a professional video script writer. Your task is to create a compelling video script divided into short verses for text-to-speech (TTS) audio generation.

CRITICAL REQUIREMENTS:
1. Each verse must describe ONLY ONE SCENE - never multiple scenes in one verse
2. Each verse should be 1-2 sentences long (10-20 words maximum)
3. Each verse should be visually descriptive for image generation
4. Create 8-12 verses total for a complete short video
5. Each verse should flow naturally into the next
6. Focus on visual elements that can be easily converted to images
7. Use present tense for better TTS delivery
8. Make each scene distinct and memorable

Format your response as a Python list of strings, where each string is one verse.
Example format: ["A dark forest at midnight with tall pine trees", "A small cabin with a warm light in the window", ...]

Do not include any explanations or markdown formatting, just the list of verses."""

        user_prompt = f"Create a video script about: {story_prompt}"
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Generate the script using OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract the generated script
        script_text = response.choices[0].message.content.strip()
        
        # Parse the script into a list
        # Remove any markdown formatting and clean up
        script_text = script_text.replace("```python", "").replace("```", "").strip()
        
        # Try to evaluate as a Python list
        try:
            script_verses = eval(script_text)
            if isinstance(script_verses, list):
                print(f"✅ Generated {len(script_verses)} script verses")
                return script_verses
            else:
                raise ValueError("Response is not a list")
        except:
            # If eval fails, try to parse manually
            # Remove brackets and split by quotes
            script_text = script_text.strip("[]")
            verses = []
            current_verse = ""
            in_quotes = False
            
            for char in script_text:
                if char == '"' and not in_quotes:
                    in_quotes = True
                elif char == '"' and in_quotes:
                    in_quotes = False
                    if current_verse.strip():
                        verses.append(current_verse.strip())
                        current_verse = ""
                elif in_quotes:
                    current_verse += char
            
            if verses:
                print(f"✅ Generated {len(verses)} script verses")
                return verses
            else:
                raise ValueError("Could not parse script verses")
        
    except Exception as e:
        print(f"Error generating video script: {e}")
        return None

def generate_detailed_image_prompts(script_verses):
    """
    Generate detailed image prompts from basic script verses to ensure character and style consistency.
    
    Args:
        script_verses (list): List of basic script verses
    
    Returns:
        list: List of detailed image prompts for consistent character generation
    """
    try:
        # Load OpenAI API key
        api_key = load_openai_api_key()
        
        # Set the API key for OpenAI
        openai.api_key = api_key
        
        print(f"Generating detailed image prompts for {len(script_verses)} verses...")
        
        # Create a detailed prompt for image prompt generation
        system_prompt = """You are a professional image prompt engineer specializing in character consistency and visual storytelling. Your task is to transform basic script verses into detailed, self-contained image prompts where each prompt fully describes the scene and characters without relying on previous context.

CRITICAL REQUIREMENTS:
1. Each detailed prompt must be 2-3 sentences long (40-60 words)
2. Each prompt must be COMPLETELY SELF-CONTAINED - describe characters fully in every prompt
3. Include specific character descriptions in EVERY prompt (age, gender, clothing, appearance, distinctive features)
4. Include detailed scene descriptions (lighting, colors, composition, mood, setting)
5. Maintain consistency in character appearance across all prompts (same character = same description)
6. Include specific visual style elements (art style, color palette, lighting)
7. Make each scene visually distinct but character-consistent
8. Focus on cinematic and professional image generation
9. Include camera angles and composition details when relevant
10. NEVER use pronouns like "he", "she", "the boy" - always describe the character fully

CHARACTER CONSISTENCY EXAMPLES:
- If first prompt mentions "a young boy named Tom with green eyes and brown hair"
- Then second prompt should say "a young boy with green eyes and brown hair opens a book" (NOT just "Tom opens a book")
- Each prompt must describe the character completely for image generation

Format your response as a Python list of strings, where each string is one detailed image prompt.
Example format: ["A young boy with green eyes and brown hair wearing a blue shirt stands in a dark Victorian mansion hallway, dramatic shadows cast by flickering candlelight, cinematic composition", "A young boy with green eyes and brown hair wearing a blue shirt opens an ancient book on a wooden table, warm candlelight illuminating the pages, atmospheric horror lighting", ...]

Do not include any explanations or markdown formatting, just the list of detailed prompts."""

        # Create the user prompt with all verses
        verses_text = "\n".join([f"{i+1}. {verse}" for i, verse in enumerate(script_verses)])
        user_prompt = f"Transform these basic script verses into detailed image prompts for consistent character generation:\n\n{verses_text}"
        
        # Generate the detailed prompts using OpenAI
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        # Extract the generated detailed prompts
        prompts_text = response.choices[0].message.content.strip()
        
        # Parse the prompts into a list
        # Remove any markdown formatting and clean up
        prompts_text = prompts_text.replace("```python", "").replace("```", "").strip()
        
        # Try to evaluate as a Python list
        try:
            detailed_prompts = eval(prompts_text)
            if isinstance(detailed_prompts, list):
                print(f"✅ Generated {len(detailed_prompts)} detailed image prompts")
                return detailed_prompts
            else:
                raise ValueError("Response is not a list")
        except:
            # If eval fails, try to parse manually
            # Remove brackets and split by quotes
            prompts_text = prompts_text.strip("[]")
            prompts = []
            current_prompt = ""
            in_quotes = False
            
            for char in prompts_text:
                if char == '"' and not in_quotes:
                    in_quotes = True
                elif char == '"' and in_quotes:
                    in_quotes = False
                    if current_prompt.strip():
                        prompts.append(current_prompt.strip())
                        current_prompt = ""
                elif in_quotes:
                    current_prompt += char
            
            if prompts:
                print(f"✅ Generated {len(prompts)} detailed image prompts")
                return prompts
            else:
                raise ValueError("Could not parse detailed prompts")
        
    except Exception as e:
        print(f"Error generating detailed image prompts: {e}")
        return None

def generate_audio_from_verse(verse, audio_filename):
    """
    Generate audio from a script verse using Kokoro TTS
    
    Args:
        verse (str): The script verse to convert to audio
        audio_filename (str): Local filename to save the audio
    
    Returns:
        str: Path to the saved audio file, or None if failed
    """
    try:
        # Load FAL API key
        api_key = load_fal_api_key()
        
        # Set the API key for fal_client
        fal_client.api_key = api_key
        
        print(f"Generating audio for: '{verse[:50]}...'")
        
        # Submit the TTS request
        endpoint = "fal-ai/kokoro/american-english"
        arguments = {
            "text": verse
        }
        
        print(f"Submitting TTS request to {endpoint}...")
        request_handle = fal_client.submit(endpoint, arguments=arguments)
        request_id = request_handle.request_id
        print(f"TTS request submitted with ID: {request_id}")
        
        # Poll for completion
        print("Waiting for audio generation to complete...")
        import time
        
        max_attempts = 30  # Maximum 1 minute of waiting for TTS
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            print(f"TTS Status Check #{attempt}")
            
            try:
                # Check status
                status = fal_client.status(endpoint, request_id, with_logs=True)
                status_type = type(status).__name__
                print(f"TTS Status: {status_type}")
                
                if status_type == "Completed":
                    print("✅ Audio generation completed!")
                    break
                elif status_type == "Failed":
                    print(f"❌ Audio generation failed")
                    if hasattr(status, 'error') and status.error:
                        print(f"Error: {status.error}")
                    return None
                elif status_type == "InProgress":
                    print("⏳ Audio generation in progress...")
                elif status_type == "Queued":
                    print("⏳ TTS request queued, waiting to start...")
                else:
                    print(f"Unknown TTS status: {status_type}")
                
                # Wait before next check
                time.sleep(2)
                
            except Exception as status_error:
                print(f"Error checking TTS status: {status_error}")
                time.sleep(2)
                continue
        
        if attempt >= max_attempts:
            print("❌ Timeout waiting for audio generation to complete")
            return None
        
        # Get the final result
        print("Getting TTS result...")
        try:
            result = fal_client.result(endpoint, request_id)
            print(f"TTS Result received: {type(result)}")
        except Exception as result_error:
            print(f"Error getting TTS result: {result_error}")
            return None
        
        # Extract the audio URL from the result
        if result and "audio" in result and "url" in result["audio"]:
            audio_url = result["audio"]["url"]
            print(f"✅ Audio generated successfully!")
            print(f"Audio URL: {audio_url}")
            
            # Download and save the audio
            saved_path = download_and_save_audio(audio_url, audio_filename)
            return saved_path
        else:
            print("❌ No audio was generated in the response")
            return None
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def download_and_save_audio(audio_url, filename):
    """
    Download an audio file from URL and save it locally
    
    Args:
        audio_url (str): URL of the audio to download
        filename (str): Local filename to save the audio
    
    Returns:
        str: Path to the saved audio file, or None if failed
    """
    try:
        print(f"Downloading audio: {filename}")
        response = requests.get(audio_url)
        response.raise_for_status()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the audio
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Audio saved: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error downloading audio {filename}: {e}")
        return None

def download_and_save_image(image_url, filename):
    """
    Download an image from URL and save it locally
    
    Args:
        image_url (str): URL of the image to download
        filename (str): Local filename to save the image
    
    Returns:
        str: Path to the saved image, or None if failed
    """
    try:
        print(f"Downloading image: {filename}")
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save the image
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Image saved: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error downloading image {filename}: {e}")
        return None

def get_next_video_folder():
    """
    Get the next available video folder number
    
    Returns:
        str: Path to the next video folder
    """
    videos_base = "videos"
    os.makedirs(videos_base, exist_ok=True)
    
    # Find the next available folder number
    folder_number = 1
    while True:
        folder_path = os.path.join(videos_base, str(folder_number))
        if not os.path.exists(folder_path):
            break
        folder_number += 1
    
    return folder_path

def generate_all_images_with_character_consistency(script_verses, style_images=None, video_folder=None, aspect_ratio="16:9"):
    """
    Generate images for all script verses, maintaining character consistency.
    For the first image, use style_images if available.
    For subsequent images, use all previously generated images as style references.
    
    Args:
        script_verses (list): List of script verses to generate images for
        style_images (list, optional): Initial style images to use for the first image
        video_folder (str, optional): Video folder path (if None, will create new one)
    
    Returns:
        list: List of paths to generated images, or None if failed
    """
    if not script_verses:
        print("No script verses provided")
        return None
    
    print(f"\n=== Generating {len(script_verses)} Images with Character Consistency ===")
    
    generated_images = []
    
    # Use provided video folder or create new one
    if video_folder is None:
        video_folder = get_next_video_folder()
        print(f"Created new video folder: {video_folder}")
    else:
        print(f"Using existing video folder: {video_folder}")
    
    images_folder = os.path.join(video_folder, "images")
    
    # Create the folder structure
    os.makedirs(images_folder, exist_ok=True)
    
    print(f"Images will be saved in: {images_folder}")
    
    for i, verse in enumerate(script_verses, 1):
        print(f"\n--- Generating Image {i}/{len(script_verses)} ---")
        print(f"Verse: {verse}")
        
        # Determine style images to use
        if i == 1:
            # First image: use provided style_images if available
            current_style_images = style_images
            print("Using initial style images for first image (style inspiration only)")
        else:
            # Subsequent images: use only previously generated images (no initial style images)
            current_style_images = generated_images
            print(f"Using {len(generated_images)} previously generated images for character consistency (no initial style images)")
        
        # Generate the image
        image_url = generate_image_based_on_style_image(current_style_images, verse, aspect_ratio)
        
        if image_url:
            # Download and save the image
            filename = os.path.join(images_folder, f"scene_{i:02d}.png")
            saved_path = download_and_save_image(image_url, filename)
            
            if saved_path:
                generated_images.append(saved_path)
                print(f"✅ Image {i} generated and saved successfully")
            else:
                print(f"❌ Failed to save image {i}")
                return None
        else:
            print(f"❌ Failed to generate image {i}")
            return None
    
    print(f"\n🎉 All {len(script_verses)} images generated successfully!")
    print(f"Video folder: {video_folder}")
    print(f"Images saved in: {images_folder}/")
    return generated_images

def generate_all_audio_files(script_verses, video_folder):
    """
    Generate audio files for all script verses using Kokoro TTS
    
    Args:
        script_verses (list): List of script verses to convert to audio
        video_folder (str): Path to the video folder where audio will be saved
    
    Returns:
        list: List of paths to generated audio files, or None if failed
    """
    if not script_verses:
        print("No script verses provided for audio generation")
        return None
    
    print(f"\n=== Generating {len(script_verses)} Audio Files ===")
    
    generated_audio = []
    audio_folder = os.path.join(video_folder, "audio")
    
    # Create the audio folder
    os.makedirs(audio_folder, exist_ok=True)
    print(f"Audio files will be saved in: {audio_folder}")
    
    for i, verse in enumerate(script_verses, 1):
        print(f"\n--- Generating Audio {i}/{len(script_verses)} ---")
        print(f"Verse: {verse}")
        
        # Generate audio filename
        audio_filename = os.path.join(audio_folder, f"scene_{i:02d}.wav")
        
        # Generate the audio
        saved_audio_path = generate_audio_from_verse(verse, audio_filename)
        
        if saved_audio_path:
            generated_audio.append(saved_audio_path)
            print(f"✅ Audio {i} generated and saved successfully")
        else:
            print(f"❌ Failed to generate audio {i}")
            return None
    
    print(f"\n🎉 All {len(script_verses)} audio files generated successfully!")
    print(f"Audio files saved in: {audio_folder}/")
    return generated_audio

def create_scene_video(image_path, audio_path, output_path, duration=None):
    """
    Create a video clip from an image and audio file using FFmpeg
    
    Args:
        image_path (str): Path to the image file
        audio_path (str): Path to the audio file
        output_path (str): Path where the video will be saved
        duration (float, optional): Duration of the video (if None, uses audio duration)
    
    Returns:
        str: Path to the created video file, or None if failed
    """
    try:
        print(f"Creating video: {os.path.basename(output_path)}")
        
        # Get audio duration using ffprobe
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', audio_path
        ]
        
        try:
            result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            audio_duration = float(result.stdout.strip())
        except:
            print(f"Warning: Could not get audio duration, using default 5 seconds")
            audio_duration = 5.0
        
        # Use provided duration or audio duration
        video_duration = duration if duration is not None else audio_duration
        
        # Create video using FFmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-loop', '1',    # Loop the image
            '-i', image_path,  # Input image
            '-i', audio_path,  # Input audio
            '-c:v', 'libx264',  # Video codec
            '-c:a', 'aac',      # Audio codec
            '-t', str(video_duration),  # Duration
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-shortest',  # End when shortest input ends
            output_path
        ]
        
        # Run FFmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video created: {output_path}")
            return output_path
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
    except Exception as e:
        print(f"Error creating video {output_path}: {e}")
        return None

def create_final_video(video_folder, scene_videos):
    """
    Combine all scene videos into one final video using FFmpeg
    
    Args:
        video_folder (str): Path to the video folder
        scene_videos (list): List of paths to individual scene videos
    
    Returns:
        str: Path to the final combined video, or None if failed
    """
    try:
        print(f"\n=== Combining {len(scene_videos)} scene videos into final video ===")
        
        if not scene_videos:
            print("No video clips to combine")
            return None
        
        # Create a temporary file list for FFmpeg
        file_list_path = os.path.join(video_folder, "file_list.txt")
        with open(file_list_path, 'w') as f:
            for video_path in scene_videos:
                print(f"Adding: {os.path.basename(video_path)}")
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        # Output path for final video
        final_video_path = os.path.join(video_folder, "final_video.mp4")
        
        # FFmpeg command to concatenate videos
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-f', 'concat',  # Use concat demuxer
            '-safe', '0',    # Allow unsafe file paths
            '-i', file_list_path,  # Input file list
            '-c', 'copy',    # Copy streams without re-encoding
            final_video_path
        ]
        
        print(f"Creating final video: {final_video_path}")
        
        # Run FFmpeg command
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        # Clean up temporary file
        if os.path.exists(file_list_path):
            os.remove(file_list_path)
        
        if result.returncode == 0:
            print(f"✅ Final video created: {final_video_path}")
            return final_video_path
        else:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
    except Exception as e:
        print(f"Error creating final video: {e}")
        return None

def create_video_from_scenes(video_folder, num_scenes):
    """
    Create individual scene videos and combine them into a final video
    
    Args:
        video_folder (str): Path to the video folder containing audio and images
        num_scenes (int): Number of scenes to process
    
    Returns:
        str: Path to the final combined video, or None if failed
    """
    try:
        print(f"\n=== Creating Video from {num_scenes} Scenes ===")
        
        # Create videos folder
        videos_folder = os.path.join(video_folder, "videos")
        os.makedirs(videos_folder, exist_ok=True)
        
        scene_videos = []
        
        # Create individual scene videos
        for i in range(1, num_scenes + 1):
            scene_num = f"{i:02d}"
            
            # Paths for this scene
            image_path = os.path.join(video_folder, "images", f"scene_{scene_num}.png")
            audio_path = os.path.join(video_folder, "audio", f"scene_{scene_num}.wav")
            video_path = os.path.join(videos_folder, f"scene_{scene_num}.mp4")
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                continue
            if not os.path.exists(audio_path):
                print(f"❌ Audio not found: {audio_path}")
                continue
            
            print(f"\n--- Creating Scene {i}/{num_scenes} ---")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Audio: {os.path.basename(audio_path)}")
            
            # Create the scene video
            created_video = create_scene_video(image_path, audio_path, video_path)
            
            if created_video:
                scene_videos.append(created_video)
                print(f"✅ Scene {i} video created successfully")
            else:
                print(f"❌ Failed to create scene {i} video")
        
        if not scene_videos:
            print("❌ No scene videos were created")
            return None
        
        # Create final combined video
        final_video_path = create_final_video(video_folder, scene_videos)
        
        if final_video_path:
            print(f"\n🎉 Video creation completed successfully!")
            print(f"Final video: {final_video_path}")
            print(f"Individual scene videos saved in: {videos_folder}/")
            return final_video_path
        else:
            print("❌ Failed to create final video")
            return None
            
    except Exception as e:
        print(f"Error creating video from scenes: {e}")
        return None

def load_style_image_if_available():
    """
    Load all image files from the /styles folder if they exist.
    Returns a list of image file paths, or None if no images are found.
    
    Supported image formats: jpg, jpeg, png, gif, bmp, tiff, webp
    """
    styles_folder = "styles"
    
    # Check if styles folder exists
    if not os.path.exists(styles_folder):
        print(f"Styles folder '{styles_folder}' does not exist.")
        return None
    
    # Define supported image file extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    
    # Find all image files in the styles folder
    image_files = []
    for extension in image_extensions:
        # Search for files with current extension (case insensitive)
        pattern = os.path.join(styles_folder, extension)
        image_files.extend(glob.glob(pattern))
        # Also search for uppercase extensions
        pattern_upper = os.path.join(styles_folder, extension.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No image files found in '{styles_folder}' folder.")
        return None
    
    print(f"Found {len(image_files)} image file(s) in '{styles_folder}' folder:")
    for img_file in image_files:
        print(f"  - {img_file}")
    
    return image_files

def upload_image_to_fal(image_path):
    """
    Upload a local image file to FAL and return the URL
    """
    try:
        print(f"Uploading image to FAL: {image_path}")
        url = fal_client.upload_file(image_path)
        print(f"✅ Image uploaded successfully: {url}")
        return url
    except Exception as e:
        print(f"Error uploading image {image_path}: {e}")
        return None

def on_queue_update(update):
    """
    Callback function for queue updates during image generation
    """
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

def generate_image_based_on_style_image(style_images, prompt, aspect_ratio="16:9"):
    """
    Generate an image using nano banana API with optional style images.
    
    Args:
        style_images (list or None): List of style image file paths, or None if no style images
        prompt (str): Text prompt for image generation
        aspect_ratio (str): Aspect ratio for the generated image (e.g., "16:9", "9:16")
    
    Returns:
        str: URL of the generated image, or None if generation failed
    """
    try:
        # Load FAL API key and set environment variable
        api_key = load_fal_api_key()
        
        print(f"Generating image with prompt: {prompt}")
        
        # Determine which API endpoint to use and prepare arguments
        if style_images and len(style_images) > 0:
            # Use style images if available
            print(f"Using {len(style_images)} style image(s)")
            
            # Upload style images to FAL
            image_urls = []
            for style_image_path in style_images:
                print(f"Uploading style image: {style_image_path}")
                image_url = upload_image_to_fal(style_image_path)
                if image_url:
                    image_urls.append(image_url)
                else:
                    print(f"Failed to upload {style_image_path}")
            
            if not image_urls:
                print("No style images were successfully uploaded, falling back to text-only generation")
                # Fall back to text-only generation
                endpoint = "fal-ai/nano-banana"
                arguments = {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio
                }
            else:
                # Use nano-banana/edit with style images and styling prompt
                endpoint = "fal-ai/nano-banana/edit"
                # Add styling instruction to the prompt - style inspiration only, no character copying
                styled_prompt = f"{prompt}, use the reference images only for visual style inspiration (colors, lighting, art style, mood) but do not copy any characters or objects from them, create completely new content"
                arguments = {
                    "prompt": styled_prompt,
                    "image_urls": image_urls,
                    "aspect_ratio": aspect_ratio
                }
        else:
            # Generate without style images using nano-banana
            print("Generating image without style reference")
            endpoint = "fal-ai/nano-banana"
            arguments = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio
            }
        
        # Submit the request
        print(f"Submitting request to {endpoint}...")
        request_handle = fal_client.submit(endpoint, arguments=arguments)
        print(f"Request submitted with handle: {request_handle}")
        
        # Extract request_id from the handle
        request_id = request_handle.request_id
        print(f"Request ID: {request_id}")
        
        # Poll for completion
        print("Waiting for image generation to complete...")
        import time
        
        max_attempts = 60  # Maximum 2 minutes of waiting
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Status Check #{attempt} ---")
            
            try:
                # Check status
                status = fal_client.status(endpoint, request_id, with_logs=True)
                status_type = type(status).__name__
                print(f"Status: {status_type}")
                
                # Show logs if available
                if hasattr(status, 'logs') and status.logs:
                    print("Logs:")
                    for log in status.logs:
                        print(f"  - {log.get('message', 'No message')}")
                
                # Check status type instead of status attribute
                if status_type == "Completed":
                    print("✅ Image generation completed!")
                    break
                elif status_type == "Failed":
                    print(f"❌ Image generation failed")
                    if hasattr(status, 'error') and status.error:
                        print(f"Error: {status.error}")
                    return None
                elif status_type == "InProgress":
                    print("⏳ Image generation in progress...")
                elif status_type == "Queued":
                    print("⏳ Request queued, waiting to start...")
                elif status_type == "NotFound":
                    print("⚠️ Request not found, waiting for it to appear...")
                else:
                    print(f"Unknown status type: {status_type}")
                
                # Wait before next check
                time.sleep(2)
                
            except Exception as status_error:
                print(f"Error checking status: {status_error}")
                time.sleep(2)
                continue
        
        if attempt >= max_attempts:
            print("❌ Timeout waiting for image generation to complete")
            return None
        
        # Get the final result
        print("\n--- Getting Final Result ---")
        try:
            result = fal_client.result(endpoint, request_id)
            print(f"Result received: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"Result attributes: {list(result.__dict__.keys())}")
        except Exception as result_error:
            print(f"Error getting result: {result_error}")
            return None
        
        # Extract the generated image URL from the result
        if result and "images" in result and len(result["images"]) > 0:
            generated_image_url = result["images"][0]["url"]
            print(f"✅ Image generation successful!")
            print(f"Generated image URL: {generated_image_url}")
            return generated_image_url
        else:
            print("❌ No image was generated in the response")
            return None
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def main():
    """
    Main function to demonstrate loading the FAL API key, style images, and image generation
    """
    print("=== Loading FAL API Key ===")
    try:
        # Load the FAL API key
        api_key = load_fal_api_key()
        
        # For security, only show first 8 characters of the API key
        masked_key = api_key[:8] + "..." if len(api_key) > 8 else api_key
        print(f"Successfully loaded FAL API key: {masked_key}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please make sure to:")
        print("1. Create a .env file in the project root")
        print("2. Add FAL_KEY=your_actual_fal_api_key to the .env file")
        print("3. Replace 'your_actual_fal_api_key' with your real FAL API key")
        return
    
    print("\n=== Loading Style Images ===")
    # Load style images
    style_images = load_style_image_if_available()
    
    if style_images:
        print(f"Style images loaded successfully: {len(style_images)} files")
    else:
        print("No style images found or styles folder doesn't exist.")
        print("To add style images:")
        print("1. Place image files (jpg, png, gif, etc.) in the 'styles' folder")
        print("2. Run the script again to load them")
    
    # COMMENTED OUT - Script, Audio, and Image Generation (already completed)
    # print("\n=== Video Script Generation Demo ===")
    # # Load story prompt from config.json
    # try:
    #     with open('config.json', 'r') as f:
    #         config = json.load(f)
    #         story_prompt = config.get('video_story_prompt', 'a mysterious forest adventure')
    #     print(f"Loaded story prompt from config.json: '{story_prompt}'")
    # except FileNotFoundError:
    #     print("config.json not found, using default story prompt")
    #     story_prompt = "a mysterious forest adventure"
    # except Exception as e:
    #     print(f"Error reading config.json: {e}, using default story prompt")
    #     story_prompt = "a mysterious forest adventure"
    # 
    # print(f"Generating video script for: '{story_prompt}'")
    # script_verses = generate_video_script_to_tts_audio(story_prompt)
    # 
    # if script_verses:
    #     print(f"✅ Script generation successful!")
    #     print(f"Generated {len(script_verses)} verses:")
    #     for i, verse in enumerate(script_verses, 1):
    #         print(f"  {i}. {verse}")
    # else:
    #     print("❌ Script generation failed. Please check your OpenAI API key and try again.")
    # 
    # print("\n=== Detailed Image Prompt Generation ===")
    # # Generate detailed image prompts for character consistency
    # if script_verses:
    #     print(f"Generating detailed image prompts for {len(script_verses)} script verses...")
    #     detailed_prompts = generate_detailed_image_prompts(script_verses)
    #     
    #     if detailed_prompts:
    #         print(f"✅ Detailed prompts generated successfully!")
    #         print(f"Generated {len(detailed_prompts)} detailed prompts:")
    #         for i, prompt in enumerate(detailed_prompts, 1):
    #             print(f"  {i}. {prompt}")
    #     else:
    #         print("❌ Detailed prompt generation failed. Using original script verses.")
    #         detailed_prompts = script_verses
    # 
    # print("\n=== Audio and Image Generation ===")
    # # Generate both audio and images in the same folder
    # if script_verses:
    #     # Get the video folder path once for both audio and images
    #     video_folder = get_next_video_folder()
    #     print(f"Using video folder: {video_folder}")
    #     
    #     print(f"\n--- Generating Audio Files ---")
    #     print(f"Generating audio files for all {len(script_verses)} script verses...")
    #     generated_audio = generate_all_audio_files(script_verses, video_folder)
    #     
    #     if generated_audio:
    #         print(f"✅ All audio files generated successfully!")
    #         print(f"Generated {len(generated_audio)} audio files:")
    #         for i, audio_path in enumerate(generated_audio, 1):
    #             print(f"  {i}. {audio_path}")
    #     else:
    #         print("❌ Audio generation failed. Please check your FAL API key and try again.")
    #     
    #     print(f"\n--- Generating Image Files ---")
    #     print(f"Generating images for all {len(script_verses)} script verses...")
    #     generated_images = generate_all_images_with_character_consistency(detailed_prompts, style_images, video_folder)
    #     
    #     if generated_images:
    #         print(f"✅ All images generated successfully!")
    #         print(f"Generated {len(generated_images)} images:")
    #         for i, image_path in enumerate(generated_images, 1):
    #             print(f"  {i}. {image_path}")
    #         
    #         print(f"\n--- Creating Final Video ---")
    #         # Create the final video from all scenes
    #         final_video_path = create_video_from_scenes(video_folder, len(script_verses))
    #         
    #         if final_video_path:
    #             print(f"🎉 Complete video creation successful!")
    #             print(f"Final video saved at: {final_video_path}")
    #         else:
    #             print("❌ Video creation failed")
    #     else:
    #         print("❌ Image generation failed. Please check your FAL API key and try again.")
    # else:
    #     print("Skipping image generation - no script verses available")

    print("\n=== VIDEO CREATION TEST ===")
    # Test video creation using existing files in videos/1/
    video_folder = "videos/1"
    
    # Check if the folder exists
    if not os.path.exists(video_folder):
        print(f"❌ Video folder not found: {video_folder}")
        print("Please make sure you have generated audio and images in videos/1/")
        return
    
    # Count existing scenes
    images_folder = os.path.join(video_folder, "images")
    audio_folder = os.path.join(video_folder, "audio")
    
    if not os.path.exists(images_folder):
        print(f"❌ Images folder not found: {images_folder}")
        return
    
    if not os.path.exists(audio_folder):
        print(f"❌ Audio folder not found: {audio_folder}")
        return
    
    # Count available scenes
    image_files = [f for f in os.listdir(images_folder) if f.startswith("scene_") and f.endswith(".png")]
    audio_files = [f for f in os.listdir(audio_folder) if f.startswith("scene_") and f.endswith(".wav")]
    
    num_scenes = min(len(image_files), len(audio_files))
    
    if num_scenes == 0:
        print("❌ No scene files found in the folders")
        return
    
    print(f"Found {num_scenes} scenes to process:")
    print(f"  Images: {len(image_files)} files")
    print(f"  Audio: {len(audio_files)} files")
    print(f"Using video folder: {video_folder}")
    
    # Create the final video from existing scenes
    print(f"\n--- Creating Final Video from Existing Files ---")
    final_video_path = create_video_from_scenes(video_folder, num_scenes)
    
    if final_video_path:
        print(f"🎉 Video creation test successful!")
        print(f"Final video saved at: {final_video_path}")
    else:
        print("❌ Video creation test failed")

if __name__ == "__main__":
    main()
