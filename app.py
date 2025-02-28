import os
from flask import Flask, render_template, request, jsonify
import torch
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load BLIP-2 and Flan-T5 Large models
caption_model = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(caption_model)
blip_model = BlipForConditionalGeneration.from_pretrained(caption_model)

t5_model = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(t5_model)
llm_model = T5ForConditionalGeneration.from_pretrained(t5_model)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to generate detailed captions
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_length=50, num_beams=5)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to extract text using OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image).strip()
    return extracted_text if extracted_text else None

# Function to generate high-quality questions
def generate_questions(text):
    prompt_text = (
        f"Generate 5 high-quality, detailed questions based on: {text}. "
        "Include questions exploring appearance, symbolism, real-world context, and scientific aspects."
    )

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    output_ids = llm_model.generate(
        input_ids, 
        max_length=150, 
        num_return_sequences=5, 
        num_beams=5, 
        temperature=0.5, 
        top_k=20
    )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No image selected"})

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
    image_file.save(image_path)

    extracted_text = extract_text_from_image(image_path)
    
    if extracted_text:
        questions = generate_questions(extracted_text)
        response = {"caption": extracted_text, "questions": questions, "image_url": image_path}
    else:
        caption = generate_caption(image_path)
        questions = generate_questions(caption)
        response = {"caption": caption, "questions": questions, "image_url": image_path}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
