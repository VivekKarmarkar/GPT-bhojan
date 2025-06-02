# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 06:58:55 2025

@author: vivek
"""

import base64
from datetime import datetime
import os
import re
import io
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from openai import OpenAI
from segment_anything import sam_model_registry, SamPredictor

# === Ensure output directory exists ===
os.makedirs("food_library", exist_ok=True)

# === Config ===
image_path = "sample_food_6.jpg"  
openai_key = "xxxx" # Replace this
client = OpenAI(api_key=openai_key)

# === Timestamp ===
timestamp = datetime.now().isoformat()

# === Load and encode image ===
with open(image_path, "rb") as f:
    image_bytes = f.read()
base64_image = base64.b64encode(image_bytes).decode("utf-8")

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

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
result = response.choices[0].message.content

# === Parse GPT Output ===
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

# === YOLO Inference ===
pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# === Create copy of original (unmodified) image ===
img_cv2_orig = img_cv2.copy()

model = YOLO("yolov8m.pt")
results = model(img_cv2)
boxes = results[0].boxes.xyxy

prompt_template = (
    "You are given a close-up image of one item from a larger food plate, along with a textual description "
    "of the full plate. Based only on the food names mentioned in the description, identify whether the image "
    "contains any of the described food items.\n\n"
    "If there's a match, respond with just the food name. If unsure or no match, respond with 'None'.\n\n"
    f"Description of full plate: {description_str} and Items: {items_str}"
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

# === SAM setup ===
sam_checkpoint = "C:/Users/naren/segment-anything/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")  # or "cpu"
predictor = SamPredictor(sam)

# Set the SAM image (RGB NumPy array)
predictor.set_image(np.array(pil_img))  # PIL image already loaded earlier
        
# === Box-wise GPT classification + SAM segmentation ===
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.tolist())
    bb_img = pil_img.crop((x1, y1, x2, y2))
    img_b64 = image_to_base64(bb_img)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }],
        max_tokens=50,
    )
    label = response.choices[0].message.content.strip()

    if label.lower() != "none":
        color = get_color(label)
        
        # === SAM with box prompt instead of center-point ===
        input_box = np.array([[x1, y1, x2, y2]])  # box format: [x1, y1, x2, y2]
        
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True
        )

        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # === Blend SAM mask on image with same color (transparency) ===
        overlay_color = np.array(color).astype(np.uint8)
        alpha = 0.6  # transparency

        img_cv2[mask] = (
            alpha * overlay_color + (1 - alpha) * img_cv2[mask]
        ).astype(np.uint8)

        # === Draw BB + Label as before ===
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=color, thickness=3)
        cv2.putText(img_cv2, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        # === Extract segmented region from original (unmodified) image ===
        segmented = np.zeros_like(img_cv2_orig)
        segmented[mask] = img_cv2_orig[mask]  # use the original image here
        
        # === Crop the bounding box region from the segmented image ===
        segmented_crop = segmented[y1:y2, x1:x2]
            
        # Convert to grayscale
        gray_crop = cv2.cvtColor(segmented_crop, cv2.COLOR_BGR2GRAY)
        
        # Create a binary mask of "non-black" pixels
        bright_pixels = gray_crop > 30  # tweak threshold if needed
        
        # Require at least N bright pixels to save
        if np.sum(bright_pixels) > 100:
            segmented_crop_rgb = cv2.cvtColor(segmented_crop, cv2.COLOR_BGR2RGB)
            Image.fromarray(segmented_crop_rgb).save(f"food_library/{label}.jpg")

# === Display the final image ===
img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img_rgb)
ax.axis("off")
plt.show()
