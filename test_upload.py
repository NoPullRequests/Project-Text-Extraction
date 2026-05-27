import requests
import json

# Test document upload
url = "http://localhost:8000/api/upload"
file_path = "uploads/invoice_sample.jpg"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response Text:")
print(response.text)
if response.status_code == 200:
    print("\nJSON Response:")
    print(json.dumps(response.json(), indent=2))
