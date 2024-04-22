# The Infilect Directory contains 2 Flask API
________________________________________________________________________________________________________________________
## Method 1:

Client.py and server.py

Navigate to Client_Server/

1. Run server.py
2. Run client.py in separate terminal
________________________________________________
Make sure to pass the necessary paths/arguments.
You can either pass --json-file (Folder of images) or --file ( single image path )
________________________________________________
Run client.py using terminal to pass the argument appropriately.

1. Example on Single Image:
    ```bash
    python client.py --file C:\Users\utkar\PycharmProjects\Infilect\Images\image-24.jpg

2. Example on JSON/Multiple Images:
    ```bash
    python client.py --json-file C:\Users\utkar\PycharmProjects\Infilect\Images\images_data.json

Note: You can create JSON file for your Image folder using create_json.py
________________________________________________________________________________________________________________________
## Method 2:
Navigate to Webapp/

Follow the steps mentioned in Webapp_Readme.txt
________________________________________________________________________________________________________________________
## Requirement

run requirements.txt
or
Download:

1. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (Download Pytorch)

2. git clone https://github.com/ultralytics/yolov5  # clone
   %cd yolov5
   pip install -r requirements.txt

3. pip install Flask
________________________________________________________________________________________________________________________
