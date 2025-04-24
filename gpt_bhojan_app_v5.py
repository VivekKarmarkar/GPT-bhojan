# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:53:44 2025

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

        # --- Parse key fields using regex (hacky but fast for now) ---
        def extract_float(pattern, default=0.0):
            match = re.search(pattern, result)
            if match:
                try:
                    return float(match.group(1))
                except:
                    return default
            return default

        health_score = extract_float(r"Health Score\*\*: (\d+\.?\d*)")
        total_calories = extract_float(r"Total Calories\*\*: (\d+\.?\d*)")
        protein = extract_float(r"protein \(g\): (\d+\.?\d*)")
        fat = extract_float(r"fat \(g\): (\d+\.?\d*)")
        carbs = extract_float(r"carbs \(g\): (\d+\.?\d*)")

        # --- Store in Supabase ---
        record = {
            "timestamp": timestamp,
            "meal_time": "unspecified",  # or infer later
            "description": "[filled by GPT output]",
            "items": "[filled by GPT output]",
            "calories": "[filled by GPT output]",
            "total_calories": total_calories,
            "health_score": health_score,
            "rationale": "[filled by GPT output]",
            "macronutrient_estimate": "[filled by GPT output]",
            "macros_protein": protein,
            "macros_fat": fat,
            "macros_carb": carbs,
            "eat_frequency": "[filled by GPT output]",
            "ideal_comparison": "[filled by GPT output]",
            "mood_impact": "[filled by GPT output]",
            "summary": "[filled by GPT output]",
            "image_url": "not stored yet"
        }

        supabase.table("food_logs").insert(record).execute()
        st.success("Logged in Supabase!")
