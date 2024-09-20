import os
import time
import google.generativeai as genai

from config import GEMINI_API_KEY

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

def transcribe_media_with_gemini(file_path):
    # Upload the media file
    print("Uploading file...")
    media_file = genai.upload_file(path=file_path)
    print(f"Completed upload: {media_file.uri}")

    # Check if the file is ready
    while media_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(10)
        media_file = genai.get_file(media_file.name)

    if media_file.state.name == "FAILED":
        raise ValueError(f"File processing failed: {media_file.state.name}")

    prompt = "Transcribe the audio, giving timestamps. Also provide visual descriptions."

    # Choose the appropriate Gemini model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Make the LLM request
    print("Making LLM inference request...")
    response = model.generate_content(
        [prompt, media_file],
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=2000,
            temperature=0.5,
        ),
        request_options={"timeout": 600},
    )

    transcript = response.text.strip()
    return transcript
