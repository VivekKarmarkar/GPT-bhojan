# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:05:38 2025

@author: vkarmarkar
"""

# GPT Bhojan + Supabase Integration

import streamlit as st
from openai import OpenAI
import base64
from datetime import datetime
from supabase import create_client, Client
import re

# --- Secrets from Streamlit ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# --- Streamlit UI Config ---
st.set_page_config(page_title="GPT Bhojan 🍛", layout="centered")
st.title("🍛 GPT Bhojan")
st.write("Upload a food photo and get insights: description, calories, health score, and more!")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    timestamp = datetime.now().isoformat()
    st.image(uploaded_file, caption="Your uploaded food plate", use_column_width=True)
    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

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

    with st.spinner("Analyzing your plate..."):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        result = response.choices[0].message.content
        st.markdown("### 🧠 Analysis:")
        st.write(result)
        
        # Regex: match number, bold heading, and grab content until the next number
        matches = re.findall(r'\d+\.\s\*\*.*?\*\*:\s*(.*?)(?=\n\d+\.|\Z)', result, re.DOTALL)
        
        description_str = matches[0]
        items_str = matches[1]
        calories_str = matches[2]
        total_calories_str = matches[3]
        health_score_str = matches[4]
        rationale_str = matches[5]
        macronutrient_estimate_str = matches[6]
        eat_frequency_str = matches[7]
        ideal_comparison_str = matches[8]
        mood_impact_str = matches[9]
        summary_str = matches[10]

        # --- Store in Supabase ---
        record = {
            "timestamp": timestamp,
            "meal_time": "unspecified",  # infer later
            "description": description_str,
            "items": items_str,
            "calories": calories_str,
            "total_calories": total_calories_str,
            "health_score": health_score_str,
            "rationale": rationale_str,
            "macronutrient_estimate": macronutrient_estimate_str,
            "macros_protein": "unspecified",
            "macros_fat": "unspecified",
            "macros_carb": "unspecified",
            "eat_frequency": eat_frequency_str,
            "ideal_comparison": ideal_comparison_str,
            "mood_impact": mood_impact_str,
            "summary": summary_str,
            "image_url": "not stored yet" # handle later
        }

        supabase.table("food_logs").insert(record).execute()
        st.success("Logged in Supabase!")