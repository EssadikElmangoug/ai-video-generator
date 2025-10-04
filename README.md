# 🎬 Shorts AI Agent - Flask Web Application

A powerful AI-powered video generation system that creates short videos with AI-generated scripts, images, and audio.

## ✨ Features

- **AI Script Generation**: Create compelling video scripts with detailed scene descriptions
- **Character-Consistent Images**: Generate images that maintain character consistency throughout your video
- **High-Quality Audio**: Generate natural-sounding voiceovers using Kokoro TTS
- **Video Assembly**: Automatically combine everything into a final MP4 video
- **Web Interface**: Beautiful, responsive web interface for easy video generation
- **Progress Tracking**: Real-time progress updates during video generation
- **Video Management**: Browse and download all your generated videos

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
sudo apt install ffmpeg
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# FAL API Key (for image and audio generation)
FAL_KEY=your_fal_api_key_here

# OpenAI API Key (for script generation)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application

```bash
# Start the Flask web application
python app.py
```

The application will be available at: `http://localhost:5000`

## 🎯 How to Use

### Generate a Video

1. **Open the web interface** at `http://localhost:5000`
2. **Enter a story prompt** (e.g., "a mysterious forest adventure" or "a horror story about a haunted house")
3. **Click "Generate Video"** and wait for the process to complete
4. **Download your video** when generation is finished

### Browse Generated Videos

- Visit the "Videos" page to see all your generated videos
- Download any video by clicking the download button
- View creation date and file size for each video

## 📁 Project Structure

```
shorts ai agent/
├── app.py                 # Flask web application
├── main.py               # Core video generation functions
├── config.json           # Story prompt configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   └── videos.html
├── static/               # Static files (CSS, JS)
├── styles/               # Style images for character consistency
├── videos/               # Generated videos (auto-created)
│   ├── 1/
│   │   ├── audio/
│   │   ├── images/
│   │   ├── videos/
│   │   └── final_video.mp4
│   └── 2/
└── venv/                 # Virtual environment
```

## 🔧 API Endpoints

- `GET /` - Main video generation page
- `POST /generate` - Start video generation
- `GET /status` - Get generation status
- `GET /videos` - List all generated videos
- `GET /download/<path>` - Download video file

## 🎨 Customization

### Style Images

Place style images in the `styles/` folder to influence the visual style of your generated videos:
- Supported formats: jpg, jpeg, png, gif, bmp, tiff, webp
- The AI will use these as style inspiration for the first image
- Subsequent images maintain character consistency

### Story Prompts

Modify `config.json` to set default story prompts:

```json
{
    "video_story_prompt": "write a horror story about a haunted house"
}
```

## 🛠️ Technical Details

### Video Generation Process

1. **Script Generation**: OpenAI GPT-3.5-turbo creates detailed script verses
2. **Image Generation**: FAL nano-banana API generates character-consistent images
3. **Audio Generation**: FAL Kokoro TTS creates natural voiceovers
4. **Video Assembly**: FFmpeg combines images and audio into final video

### Character Consistency

- First image uses style images for visual inspiration
- Subsequent images use previously generated images for character consistency
- Detailed prompts ensure character descriptions remain consistent

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg with `sudo apt install ffmpeg`
2. **API key errors**: Check your `.env` file has correct API keys
3. **MoviePy issues**: The app now uses FFmpeg directly for better compatibility
4. **Permission errors**: Ensure the app has write permissions for the videos folder

### Logs

Check the terminal where you started the Flask app for detailed error messages and progress updates.

## 📝 License

This project is for educational and personal use. Please respect the terms of service of the APIs used (OpenAI, FAL).

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Happy Video Creating! 🎬✨**
