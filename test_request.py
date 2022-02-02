import json
import requests


SERVER = 'http://127.0.0.1:8000'
headers = {'Content-Type': 'application/json'}

with open('static/data/for_test.json') as f:
    data = json.loads(f.read())

result = requests.post(SERVER,
                       headers=headers,
                       data=data)
print(result.json())
