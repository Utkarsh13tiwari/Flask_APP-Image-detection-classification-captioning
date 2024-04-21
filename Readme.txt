# The Infilect Directory contains 2 Flask APIs

## Method 1: Client.py and server.py

1. Run `server.py`.
2. Run `client.py` in a separate terminal.

**Make sure to pass the necessary paths/arguments.**
You can either pass `--json-file` (Folder of images) or `--file` (single image path).

**Run `client.py` using terminal to pass the argument appropriately.**

**Example on Single Image:**
```bash
python client.py --file path\to\single_Image

**Example on JSON/Multiple Images:**
```bash
python client.py --file path\to\json


*Note: You can create a JSON file for your Image folder using `create_json.py`.*

## Method 2: 
Navigate to `Infilect/Webapp/`.

Follow the steps mentioned in `Webapp_Readme.txt`.

## Requirements

Run `requirements.txt` or download the following:

1. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (Download PyTorch)

2. ```
git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
pip install -r requirements.txt

