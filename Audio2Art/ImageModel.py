from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa as lb
import torch
from typing import Literal
from diffusers import StableDiffusionPipeline
import time
import re
import numpy as np

def clean_transcription(text):
    """Clean up transcription text to make it more meaningful"""
    if not text or text.strip() == "":
        return "a beautiful landscape"
    
    # Remove extra spaces and clean up
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Convert to lowercase for better processing
    text = text.lower()

    # If text is too short or has no letters, fallback
    if len(text) < 3 or not any(c.isalpha() for c in text):
        return "a beautiful landscape"
    
    return text


def promptgen(file):
    """Convert audio file to text using Wav2Vec2"""
    try:
        print("Loading Wav2Vec2 model...")
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

        print("Processing audio file...")

        import tempfile, os
        if hasattr(file, 'read'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(file.read())
                file_path = tmp_file.name
            waveform, rate = lb.load(file_path, sr=16000)
            os.unlink(file_path)
        else:
            waveform, rate = lb.load(file, sr=16000)

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        elif waveform.ndim > 2:
            raise ValueError("Audio has more than 2 channels")

        if isinstance(waveform, list):
            waveform = np.array(waveform)

        if len(waveform) > 16000 * 10:
            waveform = waveform[:16000 * 10]

        if len(waveform) < 1000:
            return "a beautiful landscape"

        print(f"[DEBUG] waveform shape: {waveform.shape}, dtype: {waveform.dtype}, max: {np.max(waveform)}, min: {np.min(waveform)}")

        input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            print(f"[DEBUG] predicted_ids: {predicted_ids}")
            transcription = processor.batch_decode(predicted_ids)[0]

        print(f"[DEBUG] Raw transcription output: {transcription}")
        result = clean_transcription(transcription)
        print(f"Transcription result: {result}")
        return result

    except Exception as e:
        print(f"Error in promptgen: {e}")
        return "a beautiful landscape"


def text2image(
    prompt: str,
    repo_id: Literal[
        "dreamlike-art/dreamlike-photoreal-2.0",
        "hakurei/waifu-diffusion", 
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-2-1",
        "runwayml/stable-diffusion-v1-5",
        "nota-ai/bk-sdm-small",
        "CompVis/stable-diffusion-v1-4",
    ],
):
    """Generate image from text prompt using Stable Diffusion"""
    try:
        print(f"Generating image with prompt: {prompt}")
        print(f"Using model: {repo_id}")
        
        seed = 2024
        generator = torch.manual_seed(seed)

        if "small" in repo_id.lower():
            NUM_INFERENCE_STEPS = 15
        else:
            NUM_INFERENCE_STEPS = 20

        start = time.time()

        if torch.cuda.is_available():
            print("Using GPU")
            pipeline = StableDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to("cuda")
            pipeline.enable_attention_slicing()
            pipeline.enable_memory_efficient_attention()
        else:
            print("Using CPU")
            pipeline = StableDiffusionPipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
            pipeline.enable_attention_slicing()

        print("Generating image...")
        images = pipeline(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            num_images_per_prompt=1,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images

        end = time.time()
        print(f"Image generation completed in {end-start:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            del pipeline

        return images[0], start, end

    except Exception as e:
        print(f"Error in text2image: {e}")
        raise e