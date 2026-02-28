-- Audio Transcription & Summarization App-- 

This project is a simple web app that:
Converts speech (audio) into text and Summarizes the content into key points

It uses:

Hugging Face Transformers , Gradio for the interface
OpenAI Whisper (tiny) for speech-to-text
Mistral AI Mistral-7B for summarization

Features : 

- Upload an audio file
- Automatically transcribe speech to text (OpenAI Whisper)
- Generate concise key points from the content
- Simple web interface Using Gradio

How It Works :

- Audio is uploaded via Gradio
- Whisper model converts audio → text
- A prompt is created:
-List the key points with details 
- Mistral-7B processes the text and generates a summary

🛠️ Installation
pip install torch transformers gradio
▶️ Run the App
python app.py


💡 Future Improvements

Use a larger Whisper model for better transcription
Add support for multiple languages
Improve summarization prompts
Deploy online 
