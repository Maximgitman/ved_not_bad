import pandas as pd
import json
import requests
import os

SERVER = 'http://0.0.0.0:8080'
headers = {'Content-Type': 'application/json'}
data_path = "static/data"

test_data_path = os.path.join(data_path, 'ved_test.xlsx')
test_data = pd.read_excel(test_data_path, sheet_name=None, header=0)


test_bp = test_data['БП ']
test_bp.drop('Ответственный', inplace=True, axis=1)
orders = ['Бизнес-процесс 00-058355 от 09.12.2021 14:28:22',
          'Бизнес-процесс 00-059676 от 03.01.2022 10:28:19']
test_examples = []
for order in orders:
    test_examples.append(test_bp[test_bp['Бизнес процесс'] == order])

test_examples = pd.concat(test_examples)
test_examples.reset_index(drop=True, inplace=True)


bp_dict = test_examples.to_dict()
data = json.dumps(bp_dict, indent=4)

result = requests.post(SERVER,
    headers=headers,
    data=data)
#
print(result.status_code)
