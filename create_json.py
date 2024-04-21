"""
-----------------------------------
This script will create JSON file for your Images folder.
-----------------------------------
"""
import os
import json

# Path to the folder containing images
folder_path = r"C:\Users\utkar\Downloads\images\images"  # Pass your folder path here.

# Initialize a list to store image details
images_data = []

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Get the full path to the image
        image_path = os.path.join(folder_path, filename)

        # Append image details to the list
        images_data.append({"filename": filename, "path": image_path})

# Path to save the JSON file
json_file_path = "Images/images_data.json"

# Write the image details to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(images_data, json_file, indent=4)

print("JSON file created successfully:", json_file_path)
