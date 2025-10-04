from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import json
import threading
import time
import shutil
import io
from datetime import datetime
from main import (
    generate_video_script_to_tts_audio,
    generate_detailed_image_prompts,
    generate_all_audio_files,
    generate_all_images_with_character_consistency,
    create_video_from_scenes,
    load_style_image_if_available,
    get_next_video_folder
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Configuration for file uploads
UPLOAD_FOLDER = 'styles'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variable to track generation status
generation_status = {
    'is_running': False,
    'progress': 0,
    'current_step': '',
    'video_folder': None,
    'error': None
}

# Global variable to store uploaded style images in memory
uploaded_style_images = []

@app.route('/')
def index():
    """Main page with video generation form"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_video():
    """Start video generation process"""
    global generation_status
    
    if generation_status['is_running']:
        return jsonify({'error': 'Video generation is already in progress'}), 400
    
    # Get form data
    story_prompt = request.form.get('story_prompt', '').strip()
    
    if not story_prompt:
        return jsonify({'error': 'Story prompt is required'}), 400
    
    # Start generation in background thread
    thread = threading.Thread(target=run_video_generation, args=(story_prompt,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Video generation started', 'status': 'started'})

@app.route('/status')
def get_status():
    """Get current generation status"""
    return jsonify(generation_status)

@app.route('/upload_style', methods=['POST'])
def upload_style():
    """Upload style images to memory"""
    global uploaded_style_images
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Read file content into memory
            file_content = file.read()
            filename = secure_filename(file.filename)
            
            # Store in memory
            uploaded_style_images.append({
                'filename': filename,
                'content': file_content,
                'content_type': file.content_type
            })
            
            return jsonify({
                'message': 'Style image uploaded successfully',
                'filename': filename,
                'preview_url': f'/preview_style/{len(uploaded_style_images) - 1}'
            })
        else:
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff, webp'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preview_style/<int:index>')
def preview_style(index):
    """Serve style image preview from memory"""
    global uploaded_style_images
    
    try:
        if 0 <= index < len(uploaded_style_images):
            image_data = uploaded_style_images[index]
            return send_file(
                io.BytesIO(image_data['content']),
                mimetype=image_data['content_type']
            )
        else:
            return "Image not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated video file"""
    try:
        # Security check - only allow downloads from videos folder
        if not filename.startswith('videos/'):
            return "Access denied", 403
        
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files for playback"""
    try:
        # Security check - only allow videos from videos folder
        if not filename.startswith('videos/'):
            return "Access denied", 403
        
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/styles/<filename>')
def serve_style_image(filename):
    """Serve style images"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/videos')
def list_videos():
    """List all generated videos"""
    videos = []
    videos_dir = 'videos'
    
    if os.path.exists(videos_dir):
        for folder in sorted(os.listdir(videos_dir)):
            folder_path = os.path.join(videos_dir, folder)
            if os.path.isdir(folder_path):
                final_video = os.path.join(folder_path, 'final_video.mp4')
                if os.path.exists(final_video):
                    # Get file info
                    stat = os.stat(final_video)
                    videos.append({
                        'folder': folder,
                        'path': final_video,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                    })
    
    return render_template('videos.html', videos=videos)

def run_video_generation(story_prompt):
    """Run the complete video generation process"""
    global generation_status, uploaded_style_images
    
    try:
        generation_status.update({
            'is_running': True,
            'progress': 0,
            'current_step': 'Starting video generation...',
            'video_folder': None,
            'error': None
        })
        
        # Step 1: Generate script verses
        generation_status['current_step'] = 'Generating video script...'
        generation_status['progress'] = 10
        script_verses = generate_video_script_to_tts_audio(story_prompt)
        
        if not script_verses:
            raise Exception("Failed to generate script verses")
        
        # Step 2: Generate detailed image prompts
        generation_status['current_step'] = 'Creating detailed image prompts...'
        generation_status['progress'] = 20
        detailed_prompts = generate_detailed_image_prompts(script_verses)
        
        if not detailed_prompts:
            detailed_prompts = script_verses
        
        # Step 3: Get video folder and prepare style images
        generation_status['current_step'] = 'Preparing for generation...'
        generation_status['progress'] = 30
        video_folder = get_next_video_folder()
        
        # Convert uploaded images to temporary files for the generation function
        temp_style_images = []
        if uploaded_style_images:
            temp_dir = 'temp_styles'
            os.makedirs(temp_dir, exist_ok=True)
            for i, img_data in enumerate(uploaded_style_images):
                temp_path = os.path.join(temp_dir, f'temp_style_{i}.{img_data["filename"].split(".")[-1]}')
                with open(temp_path, 'wb') as f:
                    f.write(img_data['content'])
                temp_style_images.append(temp_path)
        else:
            # Fallback to loading from styles folder if no uploaded images
            temp_style_images = load_style_image_if_available()
        
        generation_status['video_folder'] = video_folder
        
        # Step 4: Generate audio files
        generation_status['current_step'] = 'Generating audio files...'
        generation_status['progress'] = 40
        generated_audio = generate_all_audio_files(script_verses, video_folder)
        
        if not generated_audio:
            raise Exception("Failed to generate audio files")
        
        # Step 5: Generate images
        generation_status['current_step'] = 'Generating images...'
        generation_status['progress'] = 60
        generated_images = generate_all_images_with_character_consistency(detailed_prompts, temp_style_images, video_folder)
        
        if not generated_images:
            raise Exception("Failed to generate images")
        
        # Step 6: Create final video
        generation_status['current_step'] = 'Creating final video...'
        generation_status['progress'] = 80
        final_video_path = create_video_from_scenes(video_folder, len(script_verses))
        
        if not final_video_path:
            raise Exception("Failed to create final video")
        
        # Clean up temporary files
        if uploaded_style_images and temp_style_images:
            for temp_path in temp_style_images:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            temp_dir = 'temp_styles'
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        
        # Clear uploaded images after successful generation
        uploaded_style_images.clear()
        
        # Complete
        generation_status.update({
            'is_running': False,
            'progress': 100,
            'current_step': 'Video generation completed!',
            'video_folder': video_folder,
            'error': None
        })
        
    except Exception as e:
        # Clean up temporary files on error
        if 'temp_style_images' in locals() and temp_style_images:
            for temp_path in temp_style_images:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            temp_dir = 'temp_styles'
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        
        generation_status.update({
            'is_running': False,
            'progress': 0,
            'current_step': 'Generation failed',
            'video_folder': None,
            'error': str(e)
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
