import json
import requests


SERVER = 'http://0.0.0.0:8080'
headers = {'Content-Type': 'application/json'}

with open('for_test.json') as f:
    data = json.loads(f.read())
print(data)
print(type(data))
result = requests.post(SERVER,
                       headers=headers,
                       data=data)
print(result.json())
