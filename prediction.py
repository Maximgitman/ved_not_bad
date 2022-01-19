import json

import torch
import torch.nn as nn

weights_path = 'static/model/bp_month_skills_pg_mg_64_32n.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NNModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size_1 = 64
        hidden_size_2 = 32
        self.fc_1 = nn.Linear(input_size, hidden_size_1)
        self.fc_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc_3 = nn.Linear(hidden_size_2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.bn_1 = nn.BatchNorm1d(hidden_size_1)
        self.bn_2 = nn.BatchNorm1d(hidden_size_2)

    def forward(self, input):
        x = self.relu(self.fc_1(input))
        x = self.bn_1(x)
        x = self.dropout(x)
        x = self.relu(self.fc_2(x))
        x = self.bn_2(x)
        out = self.fc_3(x)
        return out


def predict(data: object, bp_id_list: object, ved_list: object) -> object:
    model = NNModel(input_size=273).to(device)
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
