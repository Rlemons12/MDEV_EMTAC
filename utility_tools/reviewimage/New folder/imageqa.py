import os
import base64
import requests

# OpenAI API Key
api_key = "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"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to the folder containing images
folder_path = r"C:\Users\15127\OneDrive\Desktop\reviewimage"

# Define headers for the API request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Iterate through the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".png","JPG")):
        image_path = os.path.join(folder_path, filename)
        base64_image = encode_image(image_path)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        print(f"Image: {filename}")
        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                print("Response from AI:", response_data['choices'][0]['message']['content'])
            else:
                print("No response from AI.")
        else:
            print("Error in API request. Status code:", response.status_code)
