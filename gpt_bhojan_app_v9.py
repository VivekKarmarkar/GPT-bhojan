# -*- coding: utf-8 -*-
"""
@author: vkarmarkar
"""

import base64
from datetime import datetime
import re
import uuid
import io
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

import streamlit as st
from openai import OpenAI
from supabase import create_client, Client

# === Streamlit Config ===
st.set_page_config(page_title="GPT Bhojan 🍛", layout="centered")

# === API + Supabase setup ===
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# === Title and file uploader ===
st.title("🍛 GPT Bhojan")
st.write("Upload a food photo and get insights: description, calories, health score, and more!")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    timestamp = datetime.now().isoformat()

    # Create placeholder for replacing image later
    image_placeholder = st.empty()

    # Show original uploaded image
    uploaded_file.seek(0)
    image_placeholder.image(uploaded_file, caption="Your uploaded food plate", use_column_width=True)

    # Prepare image for Supabase + GPT
    uploaded_file.seek(0)
    base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")
    uploaded_file.seek(0)
    supabase.storage.from_("foodimages").upload(
        f"{uuid.uuid4()}.jpg",
        uploaded_file.getvalue(),
        {"content-type": "image/jpeg"}
    )

    image_name = f"{uuid.uuid4()}.jpg"
    image_url = f"{st.secrets['SUPABASE_URL'].replace('.supabase.co', '.supabase.co/storage/v1/object/public')}/foodimages/{image_name}"

    # === GPT Prompt ===
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
        "11. **Satiety Score**: Score from 0 to 10 based on how full this meal is likely to make the person feel.\n"
        "12. **Bloat Score**: Score from 0 to 10 based on how much bloating this meal might cause.\n"
        "13. **Tasty Score**: Score from 0 to 10 based on how tasty this meal is likely to be (based on visual and content).\n"
        "14. **Addiction Score**: Score from 0 to 10 based on how likely this meal is to trigger addictive eating patterns.\n"
        "15. **Summary**: Total calorie estimate with final health score and brief closing note."
    )

    with st.spinner("Analyzing your plate..."):
        # === GPT-4 Analysis ===
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

        # === Parse fields ===
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
        satiety_score_str = matches[10]
        bloat_score_str = matches[11]
        tasty_score_str = matches[12]
        addiction_score_str = matches[13]
        summary_str = matches[14]

        # === YOLO Setup ===
        uploaded_file.seek(0)
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        model = YOLO("yolov8m.pt")
        results = model(img_cv2)
        boxes = results[0].boxes.xyxy

        prompt_template = (
            "You are given a close-up image of one item from a larger food plate, along with a textual description "
            "of the full plate. Based only on the food names mentioned in the description, identify whether the image "
            "contains any of the described food items.\n\n"
            "If there's a match, respond with just the food name. If unsure or no match, respond with 'None'.\n\n"
            f"Description of full plate: {description_str}"
        )

        color_map = {}
        def get_color(label):
            if label not in color_map:
                color_map[label] = tuple(random.randint(100, 255) for _ in range(3))
            return color_map[label]

        def image_to_base64(image: Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        # === Loop through boxes ===
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.tolist())
            bb_img = pil_img.crop((x1, y1, x2, y2))
            img_b64 = image_to_base64(bb_img)

            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_template},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]}
                ],
                max_tokens=50,
            )
            label = response.choices[0].message.content.strip()

            if label.lower() != "none":
                color = get_color(label)
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=color, thickness=3)
                cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # === Replace image with bounding box version ===
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_rgb)
        ax.axis("off")
        image_placeholder.pyplot(fig)

        # === Log to Supabase ===
        record = {
            "timestamp": timestamp,
            "meal_time": "unspecified",
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
            "satiety_score": satiety_score_str,
            "bloat_score": bloat_score_str,
            "tasty_score": tasty_score_str,
            "addiction_score": addiction_score_str,
            "summary": summary_str,
            "image_url": image_url
        }
        supabase.table("food_logs").insert(record).execute()
        st.success("Logged in Supabase ✅")
