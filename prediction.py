import json

import numpy as np
import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = 'static/model/model_weights.pth'


class NNModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size_1 = 512
        hidden_size_2 = 256
        hidden_size_3 = 128
        self.fc_1 = nn.Linear(input_size, hidden_size_1)
        self.fc_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc_3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc_4 = nn.Linear(hidden_size_3, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn_1 = nn.BatchNorm1d(hidden_size_1)
        self.bn_2 = nn.BatchNorm1d(hidden_size_2)
        self.bn_3 = nn.BatchNorm1d(hidden_size_3)

    def forward(self, input):
        x = self.relu(self.fc_1(input))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.relu(self.fc_2(x))
        x = self.bn_2(x)
        x = self.dropout(x)
        x = self.relu(self.fc_3(x))
        x = self.bn_3(x)
        out = self.fc_4(x)
        return out


def predict(data: np.array, bp_id_list: list, ved_list: list, input_size: int) -> str:
    """
    Функция предсказания успешности выполнения заказа конкретным сотрудником ВЭД.
    Вызывает сохраненную модель нейронной сети и подает в нее данные по заказу и ВЭД.
    Возвращает предсказания по сотрудникам ВЭД для каждого переданного номера заказа.

    :param data: np.array
    :param bp_id_list: list
    :param ved_list: list
    :param input_size: int

    :return: str (json.dumps)
    """

    model = NNModel(input_size=input_size).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    bp_scores = {}
    for i, inputs in enumerate(data):
        bp_scores[bp_id_list[i]] = {}
        inputs = torch.FloatTensor(inputs).to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs.detach()).squeeze(1).cpu()
        for v, ved in enumerate(ved_list):
            bp_scores[bp_id_list[i]][ved] = round(probs[v].item(), 4)

    result = json.dumps(bp_scores, indent=4, ensure_ascii=False)
    return result


if __name__ == '__main__':
    import os
    import pandas as pd
    from preprocessing import preprocess_data

    data_path = "static/data"
    test_data_path = os.path.join(data_path, 'ved_test.xlsx')

    test_bp = pd.read_excel(test_data_path, sheet_name='БП ', header=0)
    month_kpi_skills = pd.read_excel(test_data_path, sheet_name='Характеристика ВЭД', header=1)
    quarter_kpi_skills = pd.read_excel(test_data_path, sheet_name='Характеристика ВЭД', header=1)
    positions_skills = pd.read_csv(os.path.join(data_path, "latest_positions_skills.csv"))

    test_bp.drop('Ответственный', inplace=True, axis=1)
    orders = ['Бизнес-процесс 00-058355 от 09.12.2021 14:28:22']

    test_examples = []
    for order in orders:
        test_examples.append(test_bp[test_bp['Бизнес процесс'] == order])

    test_examples = pd.concat(test_examples)
    test_examples.reset_index(drop=True, inplace=True)
    test_examples['Вид номенклатуры'] = 'Неизвестен'
    test_examples['Партнер клиента'] = 'Неизвестен'
    test_examples['Менеджер'] = 'Неизвестен'

    prepared_data, bp_id_list, ved_list = preprocess_data(test_examples,
                                                          month_kpi_skills,
                                                          quarter_kpi_skills,
                                                          positions_skills)

    input_size = prepared_data.shape[2]

    result = predict(prepared_data, bp_id_list, ved_list, input_size)
    print(result)
