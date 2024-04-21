import argparse
import requests
import json
import time

# Default URL of the Flask server
DEFAULT_SERVER_URL = "http://127.0.0.1:5000/process_image"

# Function to send a POST request to the Flask server
def send_request(url, data):
    start = time.perf_counter()
    response = requests.post(url, **data)
    end = time.perf_counter() - start
    print("Status Code:", response.status_code)
    print('{:.6f}s latency'.format(end))
    try:
        response_json = response.json()
        # Print or process the response JSON as needed
        # print("Response JSON:", response_json)
    except json.decoder.JSONDecodeError as e:
        print("Error decoding JSON:", e)

# Argument parser setup
parser = argparse.ArgumentParser(description='Send image data to Flask server')
parser.add_argument('--server-url', type=str, default=DEFAULT_SERVER_URL, help='URL of the Flask server')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--json-file', type=str, help='Path to the JSON file containing image details')
group.add_argument('--file', type=str, help='Path to a single image file')
args = parser.parse_args()

# Check if the data is a JSON file
if args.json_file:
    # Read the JSON file
    with open(args.json_file, 'r') as file:
        images_data = json.load(file)
    # Send a POST request with the image data from the JSON file
    send_request(args.server_url, {'json': images_data})
# Check if a single image file is specified
elif args.file:
    # Send a POST request with the single image file
    image_data = {'files': {'image': open(args.file, 'rb')}}
    send_request(args.server_url, image_data)
