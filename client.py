import requests
import json

url = 'http://localhost:9797/inference'
post_data = {
    'img_path': "datasets/val/0052_1664.jpg",  # 支持http/https图片url
}
response = requests.post(url, json=post_data)
print(json.loads(response.content))
