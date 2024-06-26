## Method 1 explanation:
________________________
****************************************** Client/server run locally ***************************************************

This Flask application provides an API endpoint `/process_image` to process images for object detection,
image classification, and image captioning tasks. Let's break down the approach:

1. Import Libraries: Import necessary libraries such as Flask for creating the web application, PIL for
image processing, NumPy for numerical operations, Torch for deep learning, and others.

2. Load Models: Load pre-trained deep learning models for object detection (YOLOv5), image classification(ResNet50),
and image captioning (Vision Encoder-Decoder model).

3. Define Flask App**: Initialize a Flask application.

4. Define Function to Process Single Image: Implement a function `process_single_image` to process a single image
uploaded via the API endpoint. This function performs the following steps:
   -> Load the image from the request and convert it to a NumPy array.
   -> Perform object detection using the YOLOv5 model to detect objects and their bounding boxes in the image.
   -> Extract detected regions and their labels from the object detection results.
   -> Perform image classification on the detected regions using the ResNet50 model to classify objects.
   -> Draw bounding boxes and labels on the image.
   -> Save the processed image with bounding boxes and labels.
   -> Perform image captioning using the Vision Encoder-Decoder model to generate a caption for the image.
   -> Create a JSON response containing detection results, classification results, and captioning results.

5. Define Flask Route: Define a route `/process_image` that accepts POST requests. This route calls the
`process_single_image` function to process the uploaded image and returns the JSON response.

6. Run the Flask App: Run the Flask application.

Additionally, the application also supports processing multiple images provided in JSON format. It checks if the request
is JSON and contains multiple images, then processes each image sequentially and returns the results in a list.

This approach allows users to upload single or multiple images via the API endpoint, and the application processes each
image for object detection, classification, and captioning, providing the results in a structured JSON format.

________________________________________________________________________________________________________________________

## Method 2 explanation:
________________________
******************************************************* Webapp *********************************************************
1. Uploading Images: Users can upload an image file via a POST request to the `/process_image` endpoint.

2. Image Processing: Upon receiving an image, the application performs the following tasks:
   -> Object Detection: It uses the YOLOv5 model to detect objects in the image and draws bounding boxes around them.
   -> Image Classification: Detected objects are classified using the ResNet50 model to determine their category.
   -> Image Captioning: The image is also passed through a Vision Encoder-Decoder model to generate a caption.

3. Response: The processed image with bounding boxes and labels, along with classification and captioning results,
are returned as a JSON response.

4. Rendering: Additionally, the application provides a simple HTML template (`index.html`) rendered at the root URL.

5. Running the App: When executed, the Flask app starts running locally on port 5000.

To use this application, you need to send a POST request to the `/process_image` endpoint with an image file attached.
________________________________________________________________________________________________________________________

## Improvements/other approaches:
_________________________________
1. Use batch processing: Process multiple Images into batches or use multiple threads to work individually on each Image

2. NVIDIA SDK's: Use NVIDIA SDK's such as <Deepstream>(for video's in future) and "Triton Inference server> to do
parallel inferencing/serving multiple request of users dynamically.

3. SOTA Models: To enhance image processing capabilities, we will explore state-of-the-art (SOTA) models like YOLOv9 and
others, evaluating their latency and architecture size. The focus is on assessing the tradeoff between parameter size
and accuracy. While lightweight models may reduce latency, they might not always guarantee superior accuracy.
By systematically testing these models and analyzing their performance metrics, we aim to identify the most efficient
model that offers optimal tradeoffs between latency, model size, and accuracy, ensuring it meets the application's
requirements for efficient image processing.
________________________________________________________________________________________________________________________
