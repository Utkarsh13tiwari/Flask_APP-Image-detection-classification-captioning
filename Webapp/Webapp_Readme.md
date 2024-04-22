# Image Processing Web App

This is a Flask web application that performs various image processing tasks using deep learning models. It includes functionality for object detection, image classification, and image captioning.
________________________________________________________________________________________________________________________
## Getting Started

To run the application locally, follow these steps:

1. Clone this repository to your local machine/Download the Zip file.

2. Navigate to the project directory.

3. Install the required Python packages from `requirements.txt`.

4. Run the Flask application using `python webapp.py`.

5. Open a web browser and go to `http://127.0.0.1:5000` to access the application.
________________________________________________________________________________________________________________________
## Usage

### Uploading Images

You can upload an image using the web interface by clicking the "Choose File" button and selecting an image file from your local system.

1. "Choose File"
2. Click on Upload

Result will be displayed.
________________________________________________________________________________________________________________________
### Image Processing

Once an image is uploaded, the application performs the following tasks:

-> Object Detection: Detects objects in the image using the YOLOv5 model.
-> Image Classification: Classifies objects detected in the image using a pre-trained ResNet50 model.
-> Image Captioning: Generates a caption for the uploaded image using a Vision Encoder-Decoder model.
________________________________________________________________________________________________________________________
### Viewing Results

The processed image with bounding boxes and labels, along with classification results and image captions, are displayed on the web interface.
________________________________________________________________________________________________________________________
## Technologies/Models Used

-> Flask: Web framework for building the application.
-> PyTorch: Deep learning library used for object detection, image classification, and image captioning tasks.
-> YOLOv5: Object detection model used for detecting objects in images.
-> ResNet50: Pre-trained model for image classification.
-> Vision Encoder-Decoder Model: Pre-trained model for image captioning.
________________________________________________________________________________________________________________________
# Results

-> See the Screenshots in the Webapp/Results/ Directory.
