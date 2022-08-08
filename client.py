import requests
import glob
headers = {'User-Agent': 'Mozilla/5.0'}
# img_paths = glob.glob("tmp/*")
img_paths = ["tmp/test.jpg"]
imgs_rb = []
for img_path in img_paths:
    with open(img_path, "rb") as f:
        imgs_rb.append(("imgs", f.read()))
response = requests.post('http://192.168.67.138:8123/predict/', headers=headers, files=imgs_rb)
print(response.status_code)
print(response.json())