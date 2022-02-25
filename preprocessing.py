import os
import json
import joblib

import numpy as np
import pandas as pd


data_path = "static/data"

scaler = joblib.load(os.path.join(data_path, 'scaler_state.joblib'))

with open(os.path.join(data_path, 'partner_dict.json'), 'r') as f:
    partner_dict = json.load(f)

with open(os.path.join(data_path, 'position_low_dict.json'), 'r') as f:
    position_low_dict = json.load(f)

with open(os.path.join(data_path, 'position_high_dict.json'), 'r') as f:
    position_high_dict = json.load(f)

with open(os.path.join(data_path, 'manager_dict.json'), 'r') as f:
    manager_dict = json.load(f)

with open(os.path.join(data_path, 'week_dict.json'), 'r') as f:
    week_dict = json.load(f)

month_skills_columns = ['ВЭД', 'Звонки', 'Звонки норма', 'Обработанные заявки',
                        'Норма 88%', 'Обработка не позднее 48 часов', 'Норма 85%',
                        'Полнота сбора', 'Норма 95%', 'Встречи', 'Встречи норма']

quarter_skills_columns = ['ВЭД', 'Звонки (3 мес)', 'Звонки норма (3 мес)', 'Обработанные заявки (3 мес)',
                          'Норма 88% (3 мес)', 'Обработка не позднее 48 часов (3 мес)',
                          'Норма 85% (3 мес)', 'Полнота сбора (3 мес)', 'Норма 95% (3 мес)',
                          'Встречи (3 мес)', 'Встречи норма (3 мес)']

skills_final_columns = ['ВЭД', 'Звонки', 'Обработанные заявки', 'Норма 88%',
                        'Обработка не позднее 48 часов', 'Норма 85%', 'Полнота сбора',
                        'Норма 95%', 'Встречи', 'Звонки (3 мес)', 'Звонки / Норма',
                        'Обработанные заявки (3 мес)', 'Норма 88% (3 мес)',
                        'Обработанные заявки / Норма', 'Обработка не позднее 48 часов (3 мес)',
                        'Норма 85% (3 мес)', '48 часов / Норма', 'Полнота сбора (3 мес)',
                        'Норма 95% (3 мес)', 'Полнота сбора / Норма', 'Встречи (3 мес)',
                        'Встречи / Норма']


def preprocess_bp(bp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Функция принимает на вход DataFrame заказов со столбцами:
    'Бизнес процесс', 'Группа видов номенклатуры', 'Партнер клиента', 'Менеджер'
    Возвращает преобразованный DataFrame со столбцами:
    'Бизнес процесс', 'Дата', 'Неделя', 'Партнер клиента', 'Менеджер', 'Группа видов номенклатуры'

    :param bp_data: pd.DataFrame

    :return: pd.DataFrame
    """

    bp_data['Бизнес процесс'] = bp_data['Бизнес процесс'].astype('str')

    # Заполнение пропущенных значений в номенклатурах
    bp_data['Группа видов номенклатуры'].fillna('UNK', inplace=True)
    bp_data['Вид номенклатуры'].fillna('UNK', inplace=True)

    # Извлечение id и даты БП
    order_id = []
    order_date = []
    for i in range(len(bp_data['Бизнес процесс'])):
        process = bp_data['Бизнес процесс'][i]
        order_id.append(process[15:24])
        order_date.append(process[-19:])
    bp_data['Бизнес процесс'] = order_id
    bp_data['Дата'] = order_date
    bp_data['Дата'] = pd.to_datetime(bp_data['Дата'])
    bp_data['Неделя'] = bp_data['Дата'].dt.isocalendar().week

    # Объединение номенклатур по БП
    bp_unique = bp_data['Бизнес процесс'].unique()
    unique_bp = []
    for bp in bp_unique:
        bp_data_slice = bp_data[bp_data['Бизнес процесс'] == bp]
        positions_groups = '__'.join(bp_data_slice['Группа видов номенклатуры'].unique())
        positions_species = '__'.join(bp_data_slice['Вид номенклатуры'].unique())
        new_line = bp_data_slice.iloc[0].copy()
        new_line['Группа видов номенклатуры'] = positions_groups
        new_line['Вид номенклатуры'] = positions_species
        unique_bp.append(new_line)

    bp_data_unique = pd.DataFrame(unique_bp, columns=bp_data.columns)
    bp_data_unique.reset_index(drop=True, inplace=True)

    columns = ['Бизнес процесс', 'Дата', 'Неделя', 'Партнер клиента', 'Менеджер', 'Группа видов номенклатуры',
               'Вид номенклатуры']
    bp_data_unique = bp_data_unique.reindex(columns=columns)

    return bp_data_unique


def preprocess_skills(month_kpi_skills: pd.DataFrame, quarter_kpi_skills: pd.DataFrame) -> pd.DataFrame:
    """
    Функция принимает на вход два DataFrame:
    - с данными по KPI сотрудников ВЭД за последний месяц
    - с данными по KPI сотрудников ВЭД за последний квартал
    Возвращает объединенный DataFrame по двум таблицам с дополнительными признаками отношений выполненных работ
    к нормам сотрудников

    :param month_kpi_skills: pd.DataFrame
    :param quarter_kpi_skills: pd.DataFrame

    :return: pd.DataFrame
    """

    month_kpi_skills.fillna(0, inplace=True)
    quarter_kpi_skills.fillna(0, inplace=True)

    # Переносим данные по месячным скилам в один дата-фрейм
    month_kpi_skills.columns = month_skills_columns
    quarter_kpi_skills.columns = quarter_skills_columns

    assert sorted(month_kpi_skills['ВЭД'].unique()) == sorted(quarter_kpi_skills['ВЭД'].unique()), 'В таблицах KPI за месяц из за квартал содержатся разные ВЭД'

    kpi_skills = month_kpi_skills.merge(quarter_kpi_skills, on='ВЭД', how='inner')

    # Считаем отношения между результатами за 3 мес и нормами
    kpi_skills['Звонки / Норма'] = kpi_skills['Звонки (3 мес)'] / kpi_skills['Звонки норма (3 мес)']
    kpi_skills['Обработанные заявки / Норма'] = kpi_skills['Обработанные заявки (3 мес)'] / kpi_skills['Норма 88% (3 мес)']
    kpi_skills['48 часов / Норма'] = kpi_skills['Обработка не позднее 48 часов (3 мес)'] / kpi_skills['Норма 85% (3 мес)']
    kpi_skills['Полнота сбора / Норма'] = kpi_skills['Полнота сбора (3 мес)'] / kpi_skills['Норма 95% (3 мес)']
    kpi_skills['Встречи / Норма'] = kpi_skills['Встречи (3 мес)'] / kpi_skills['Встречи норма (3 мес)']
    kpi_skills.fillna(0.0, inplace=True)  # Заполняем NaN там, где возникло деление на 0

    kpi_skills.drop(['Звонки норма', 'Встречи норма', 'Звонки норма (3 мес)', 'Встречи норма (3 мес)'], axis=1, inplace=True)
    kpi_skills = kpi_skills.reindex(columns=skills_final_columns)

    return kpi_skills


def to_ohe(feature, d):
    """
    Функция кодирования категориальных признаков в вектор OHE

    :param feature: кодируемый признак
    :param d:       словарь для кодирования признаков

    :return: vector: вектор OHE
    """
    vector = [0] * len(d)
    if feature in d:
        vector[d[feature]] = 1
    else:
        vector[d['UNK']] = 1
    return vector


def to_bow(feature, d):
    """
    Функция кодирования категориальных признаков в вектор BoW

    :param feature: кодируемый признак
    :param d:       словарь для кодирования признаков

    :return: vector: вектор BoW
    """
    vector = [0] * len(d)
    values = feature.split('__')
    for value in values:
        if value in d:
            vector[d[value]] = 1
        else:
            vector[d['UNK']] = 1
    return vector


def prepare_cat_features(data: pd.DataFrame) -> np.array:
    """
    Функция преобразования категориальных признаков.
    Принимает на вход DataFrame с данными по заказам.
    Возвращает numpy-массив с преобразованными признаками заказов

    :param data: pd.DataFrame

    :return: np.array
    """
    x_cat = []
    for row in np.array(data):
        prepared_row = []
        prepared_row += to_bow(str(row[2]), week_dict)
        prepared_row += to_ohe(row[3], partner_dict)
        prepared_row += to_ohe(row[4], manager_dict)
        prepared_row += to_bow(row[5], position_low_dict)
        prepared_row += to_bow(row[6], position_high_dict)
        x_cat.append(prepared_row)
    x_cat = np.array(x_cat, dtype=np.float32)
    return x_cat


def preprocess_data(bp_data, month_kpi_skills, quarter_kpi_skills, positions_skills):
    """
    Функция предобработки входных данных.
    Принимает на вход четыре DataFrame:
    - описание заказа/заказов
    - месячные данные по KPI
    - квартальные данные по KPI
    - исторические данные по работе каждого ВЭД с различными номенклатурами
    Возвращает:
    - преобразованные данные
    - список бизнес-процессов
    - список анализируемых сотрудников ВЭД (на основании таблицы с данными KPI)

    :param bp_data: pd.DataFrame
    :param month_kpi_skills: pd.DataFrame
    :param quarter_kpi_skills: pd.DataFrame
    :param positions_skills: pd.DataFrame

    :return:
    prepared_data: np.array
    bp_id_list: list
    ved_list: list
    """

    # Преобразуем данные по заказам
    bp_data_unique = preprocess_bp(bp_data.copy())

    # Преобразум категориальные признаки по заказам
    bp_cat_features = prepare_cat_features(bp_data_unique)

    # Преобразуем данные по KPI
    kpi_skills = preprocess_skills(month_kpi_skills.copy(), quarter_kpi_skills.copy())

    # Объединяем данные по KPI и исторические данные по работе ВЭД с номенклатурами
    # По ВЭД, которого нет в таблице с историческими данными, его навыки по номенклатурам заполняются нулями
    skills = kpi_skills.merge(positions_skills, on='ВЭД', how='left')
    skills.drop(['Год', 'Месяц'], axis=1, inplace=True)
    skills.fillna(0.0, inplace=True)

    # Преобразуем числовые признаки в характеристиках ВЭД
    num_features = scaler.transform(skills.iloc[:, 1:])

    # Объединяем все заказы со всеми ВЭД
    prepared_data = []
    for bp in bp_cat_features:
        prepared_data.append(np.hstack((num_features, bp.reshape((1, -1)).repeat(skills.shape[0], axis=0))))
    prepared_data = np.asarray(prepared_data)

    bp_id_list = bp_data_unique['Бизнес процесс'].to_list()
    ved_list = skills['ВЭД'].to_list()

    return prepared_data, bp_id_list, ved_list


if __name__ == '__main__':
    test_data_path = os.path.join(data_path, 'ved_test.xlsx')

    test_bp = pd.read_excel(os.path.join(data_path, 'ved_test.xlsx'), sheet_name='БП ', header=0)
    month_kpi_skills = pd.read_excel(os.path.join(data_path, 'month_kpi_skills.xlsx'),
                                     sheet_name='Характеристика ВЭД', header=1)
    quarter_kpi_skills = pd.read_excel(os.path.join(data_path, 'quarter_kpi_skills.xlsx'),
                                       sheet_name='Характеристика ВЭД', header=1)
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

    print(prepared_data.shape)
    print(bp_id_list)
    print(ved_list)
