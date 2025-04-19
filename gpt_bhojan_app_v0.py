# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 14:37:08 2025

@author: vkarmarkar
"""

import streamlit as st
from openai import OpenAI
import base64

# ✅ Securely load your API key from secrets.toml
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🧠 Streamlit app UI
st.set_page_config(page_title="GPT Bhojan 🍛", layout="centered")
st.title("🍛 GPT Bhojan")
st.write("Upload a photo of your plate, and I'll tell you what's on it and how many calories it might have.")

# 📷 Upload image
uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Your uploaded food plate", use_column_width=True)

    # Convert to base64
    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    # Send image to GPT-4 Vision
    with st.spinner("Analyzing your plate..."):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the food in this image and give a rough calorie estimate as well as total calorie estimate."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        # Show response
        result = response.choices[0].message.content
        st.markdown("### 🍽️ Result:")
        st.write(result)
