import streamlit as st
from ImageModel import promptgen, text2image
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def app():
    st.title("Audio2Art: Transforming Audio Prompts into Visual Creations")
    
    # Add some styling
    st.markdown("""
    <style>
    .stSpinner > div > div {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    upload_file = st.file_uploader("Choose your .wav audio file", type=["wav"])
    
    option = st.selectbox(
        "Select model for Image Generation:",
        (
            "nota-ai/bk-sdm-small",  # Fastest option first
            "CompVis/stable-diffusion-v1-4", 
            "runwayml/stable-diffusion-v1-5",
            "prompthero/openjourney",
            "hakurei/waifu-diffusion",
            "stabilityai/stable-diffusion-2-1",
            "dreamlike-art/dreamlike-photoreal-2.0",
        ),
    )
    
    # Add model information
    model_info = {
        "nota-ai/bk-sdm-small": "⚡ Fastest generation (~30-60s)",
        "CompVis/stable-diffusion-v1-4": "🎨 Classic Stable Diffusion",
        "runwayml/stable-diffusion-v1-5": "🔥 Popular and reliable",
        "prompthero/openjourney": "🌟 Artistic style",
        "hakurei/waifu-diffusion": "🎭 Anime/cartoon style",
        "stabilityai/stable-diffusion-2-1": "🚀 Latest Stability AI",
        "dreamlike-art/dreamlike-photoreal-2.0": "📸 Photorealistic results"
    }
    
    st.info(f"Selected: {model_info.get(option, 'Standard model')}")
    
    if st.button("🎵 Generate Image from Audio!", type="primary"):
        if upload_file is not None:
            # Display file info
            st.write(f"**File:** {upload_file.name}")
            st.write(f"**Size:** {upload_file.size} bytes")
            
            with st.spinner(text="🎧 Processing audio and generating image... This may take 1-3 minutes."):
                try:
                    # Reset file pointer to beginning
                    upload_file.seek(0)
                    
                    # Generate prompt from audio
                    st.write("🔄 Step 1: Converting speech to text...")
                    prompt = promptgen(upload_file)
                    
                    # Show the generated prompt
                    st.success(f"✅ **Generated prompt:** {prompt}")
                    
                    # Generate image from prompt
                    st.write("🔄 Step 2: Generating image from text...")
                    im, start, end = text2image(prompt, option)
                    
                    # Prepare image for download
                    buf = BytesIO()
                    im.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    # Calculate processing time
                    hours, rem = divmod(end - start, 3600)
                    minutes, seconds = divmod(rem, 60)
                    
                    st.success(
                        f"🎉 **Processing completed!** Time taken: {int(minutes):02d}:{seconds:05.2f}"
                    )
                    
                    # Display image
                    st.image(im, caption=f"Generated from: '{prompt}'", use_column_width=True)
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Generated Image",
                        data=byte_im,
                        file_name=f"audio2art_{prompt[:20].replace(' ', '_')}.png",
                        mime="image/png",
                        type="secondary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.error("Please check that your audio file is a valid .wav file and try again.")
        else:
            st.warning("⚠️ Please upload an audio file first.")
    
    # Sidebar
    st.sidebar.markdown("## 📖 Guide")
    st.sidebar.info("This tool uses Wav2Vec2 and Stable Diffusion to convert your audio prompt into an image!")
    
    st.sidebar.markdown("### 🎯 How to use:")
    st.sidebar.write("1. 🎤 Record or upload a .wav audio file")
    st.sidebar.write("2. 🎨 Choose an AI model")
    st.sidebar.write("3. 🚀 Click generate and wait")
    st.sidebar.write("4. 💾 Download your created image!")
    
    st.sidebar.markdown("### 💡 Tips:")
    st.sidebar.write("- Speak clearly and describe what you want to see")
    st.sidebar.write("- Use descriptive words like 'beautiful', 'colorful', 'detailed'")
    st.sidebar.write("- Keep audio under 10 seconds for best results")
    
    st.sidebar.markdown("### ⚡ Models:")
    st.sidebar.write("- **bk-sdm-small**: Fastest (~30-60s)")
    st.sidebar.write("- **stable-diffusion-v1-5**: Best balance")
    st.sidebar.write("- **dreamlike-photoreal**: Most realistic")

if __name__ == "__main__":
    app()