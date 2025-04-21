# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:37:20 2025

@author: vkarmarkar
"""

import streamlit as st
from openai import OpenAI
import base64
from datetime import datetime

# ✅ Load OpenAI API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🧠 UI config
st.set_page_config(page_title="GPT Bhojan 🍛", layout="centered")
st.title("🍛 GPT Bhojan")
st.write("Upload a food photo and get insights: description, calories, health score, and more!")

# 📸 Upload image
uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ⏱ Timestamp (local time)
    timestamp = datetime.now().isoformat()

    # 🖼 Show image
    st.image(uploaded_file, caption="Your uploaded food plate", use_column_width=True)

    # 🔄 Convert image to base64
    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    # 💬 Prompt
    prompt = (
        "You are GPT Bhojan 🍛, a food and nutrition assistant.\n\n"
        "Please analyze the food in this image and return a structured analysis in this format:\n\n"
        "1. **Description**: A short paragraph describing the food.\n"
        "2. **Items**: A list of distinct items on the plate.\n"
        "3. **Calories**: Estimate calories for each item and the total.\n"
        "4. **Total Calories**: Tell me the total calorie estimate.\n"
        "5. **Health Score**: Give a score from 0 to 10 (real number).\n"
        "6. **Rationale**: Explain why this score was given.\n"
        "7. **Macronutrient Estimate**: Rough protein (g), fat (g), carbs (g).\n"
        "8. **Eat Frequency**: Label as one of ['Can eat daily', 'Occasional treat', 'Avoid except rarely'].\n"
        "9. **Comparison to Ideal Meal**: Brief comment on how this compares to a typical healthy benchmark meal.\n"
        "10. **Mood/Energy Impact**: What short-term effects might this food have (e.g., energy crash, satiety)?\n"
        "11. **Summary**: Total calorie estimate with final health score for the plate"
    )

    # 🚀 Call GPT-4 Vision
    with st.spinner("Analyzing your plate..."):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }},
                    ],
                }
            ],
        )

        # 🧾 Display result
        result = response.choices[0].message.content
        st.markdown("### 🧠 Analysis:")
        st.write(result)

        # 🕒 Log summary (locally, optionally)
        st.markdown("##### ⏱ Timestamp:")
        st.code(timestamp)
