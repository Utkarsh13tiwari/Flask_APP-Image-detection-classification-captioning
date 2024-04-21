import io
import json
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template
from torchvision.models import resnet50, ResNet50_Weights
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

app = Flask(__name__)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------Load models--------------------------------
# Load classification model
weights = ResNet50_Weights.IMAGENET1K_V2
classification_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
classification_model.eval()
classification_model.to(device)
preprocess = weights.transforms()

# Load image captioning model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load detection model
detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

# Load font for label
font = ImageFont.truetype("arial.ttf", 20)

# --------------------------------------------------------------------------------------

# Function to process a single image
import base64

@app.route('/process_image', methods=['POST'])
def process_image():
    # Process single image
    response = process_single_image(request.files['image'])
    return jsonify(response)

def process_single_image(image_file):
    # Load image from request and move it to GPU
    img_pil = Image.open(io.BytesIO(image_file.read()))
    img_np = np.array(img_pil)

    # Perform detection
    results = detection_model([img_np], augment=False)

    # Draw bounding boxes and labels on the image
    img_with_boxes = img_pil.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    detected_regions = []
    detected_labels = []
    classification_results = []
    category_names = []
    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, class_id, conf = map(int, detection.tolist())
        detected_region = img_np[int(ymin):int(ymax), int(xmin):int(xmax)]
        detected_regions.append(detected_region)
        detected_labels.append(class_id)

        # Perform classification on detected regions
        region = Image.fromarray(detected_region)
        preprocessed_region = preprocess(region).unsqueeze(0).to(device)
        classification_result = classification_model(preprocessed_region).squeeze(0).softmax(0)
        class_id = classification_result.argmax().item()
        score = classification_result[class_id].item()
        category_name = weights.meta["categories"][class_id]
        category_names.append(category_name)
        classification_results.append((category_name, score))

    img_with_boxes = img_pil.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for detection, label in zip(results.xyxy[0], category_names):
        xmin, ymin, xmax, ymax, _, conf = map(int, detection.tolist())
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
        draw.text((xmin, ymin - 20), label, fill="red", font=font)

    # Save the image with drawn bounding boxes and labels
    buffered = io.BytesIO()
    img_with_boxes.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    captioning_results = []

    # Perform image captioning
    images = [img_np]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=15, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    captioning_results.append(preds)

    # Return response
    response = {
        "full_image": img_str,
        "classification_results": classification_results,
        "captioning_results": captioning_results,
    }

    return response

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
