import io
import json
import math
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify
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
def process_single_image(image_file):
    # Load image from request and move it to GPU
    img_pil = Image.open(io.BytesIO(image_file.read()))
    img_np = np.array(img_pil)

    # Perform detection
    results = detection_model([img_np], augment=False)
    # Store detected regions and their labels
    detected_regions = []
    detected_labels = []
    classification_results = []
    category_names = []
    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, class_id, _ = detection.tolist()
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

    # Draw bounding boxes and labels on the image
    img_with_boxes = img_pil.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for detection, label in zip(results.xyxy[0], category_names):
        xmin, ymin, xmax, ymax, _, conf = map(int, detection.tolist())
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
        draw.text((xmin, ymin - 20), label, fill="red", font=font)

    # Save the image with drawn bounding boxes and labels
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # To differ the names of Images.
    img_with_boxes.save(
        f"Result_Images/result_image{timestamp}.jpg")  # Images Result stored in Result_Images Directory.

    # Perform image captioning
    images = [img_np]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=15, num_beams=4)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    print("Caption: ", preds)

    # Create response JSON
    response = {
        "detection_results": [(detected_region.tolist(), label) for detected_region, label in
                              zip(detected_regions, detected_labels)],
        "classification_results": classification_results,
        "captioning_results": preds,
    }

    return response


@app.route('/process_image', methods=['POST'])
def process_image():
    if request.files:
        # Process single image
        response = process_single_image(request.files['image'])
        return jsonify(response)
    elif request.is_json:
        # Process JSON containing multiple images
        json_data = request.get_json()
        if isinstance(json_data, list):
            results_list = []
            batch_size = 3  # Define the batch size
            for i in range(0, len(json_data), batch_size):
                batch_images = json_data[i:i + batch_size]
                batch_results = []
                for image_data in batch_images:
                    filename = image_data.get('filename')
                    path = image_data.get('path')
                    if filename and path:
                        # Load image from path and process
                        with open(path, 'rb') as f:
                            image_file = io.BytesIO(f.read())
                            response = process_single_image(image_file)
                            response['filename'] = filename
                            batch_results.append(response)
                results_list.extend(batch_results)
            return jsonify(results_list)
        else:
            return "Invalid JSON format"
    else:
        return "No valid image data found"


if __name__ == '__main__':
    app.run(debug=True)
