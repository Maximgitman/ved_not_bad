import os
import json
import joblib

import numpy as np
import pandas as pd


data_path = "static/data"

scaler = joblib.load(os.path.join(data_path, 'scaler_state.joblib'))

with open(os.path.join(data_path, 'partner_dict.json'), 'r') as f:
    partner_dict = json.load(f)

with open(os.path.join(data_path, 'position_dict.json'), 'r') as f:
    position_dict = json.load(f)

with open(os.path.join(data_path, 'manager_dict.json'), 'r') as f:
    manager_dict = json.load(f)


# 1. Preprocessing all_bp (maybe will it be Json)


def preprocess_bp(bp_data: pd.DataFrame) -> pd.DataFrame:
    bp_data['Бизнес процесс'] = bp_data['Бизнес процесс'].astype('str')

    # Заполнение пропущенных значений в номенклатурах
    bp_data['Группа видов номенклатуры'].fillna('Временная', inplace=True)

    # Извлечение id и даты БП
    order_id = []
    order_date = []
    for i in range(len(bp_data['Бизнес процесс'])):
        process = bp_data['Бизнес процесс'][i]
        order_id.append(process[15:24])
        order_date.append(process[-19:])
    bp_data['Бизнес процесс'] = order_id
    bp_data['Дата БП'] = order_date

    # Объединение номенклатур по БП
    bp_unique = bp_data['Бизнес процесс'].unique()
    unique_bp = []
    for bp in bp_unique:
        bp_data_slice = bp_data[bp_data['Бизнес процесс'] == bp]
        positions = '__'.join(bp_data_slice['Группа видов номенклатуры'].unique())
        new_line = bp_data_slice.iloc[0].copy()
        new_line['Группа видов номенклатуры'] = positions
        unique_bp.append(new_line)

    bp_data_unique = pd.DataFrame(unique_bp, columns=bp_data.columns)
    bp_data_unique.reset_index(drop=True, inplace=True)

    # TODO убрать заглушки
    bp_data_unique['Партнер клиента'] = 'Неизвестен'
    bp_data_unique['Менеджер'] = 'Неизвестен'

    columns = ['Бизнес процесс', 'Дата БП', 'Партнер клиента', 'Менеджер', 'Группа видов номенклатуры']
    bp_data_unique = bp_data_unique.reindex(columns=columns)

    return bp_data_unique


# 2. Preprocessing time_process --> add to folder static/data


def preprocess_time(time: pd.DataFrame) -> pd.DataFrame:
    time['Время выполнения в секундах'] = time['Время выполнения в секундах'].astype(str).str.replace(',', '').astype(
        'float')

    order_id = []
    order_date = []
    for i in range(len(time['Бизнес-процесс'])):
        process = time['Бизнес-процесс'][i]
        order_id.append(process[15:24])
        month = process[31:38]
        order_date.append(month)
    time['Бизнес-процесс'] = order_id
    time['Дата'] = order_date

    # Рассчитаем месячную медиану по времени выполнения заказа по каждому ВЭД
    median_time = time.groupby(['ВЭД', 'Дата'], as_index=False).median()
    median_time.rename(columns={'Время выполнения в секундах': 'Медиана времени выполнения (сек)'}, inplace=True)

    return median_time


# 3. Preprocessing skills --> add to folder static/data


skills_columns = ['N', 'ВЭД', 'Звонки', 'Звонки норма',
                  'Обработанные заявки', 'Норма 88%',
                  'Обработка не позднее 48 часов', 'Норма 85%',
                  'Полнота сбора', 'Норма 95%',
                  'Встречи', 'Встречи норма', 'Дата']

columns_to_prepare = ['Звонки', 'Звонки норма', 'Обработанные заявки', 'Норма 88%',
                      'Обработка не позднее 48 часов', 'Норма 85%', 'Полнота сбора',
                      'Норма 95%', 'Встречи', 'Встречи норма']

columns_to_check = ['Звонки', 'Обработанные заявки', 'Норма 88%',
                    'Обработка не позднее 48 часов', 'Норма 85%',
                    'Полнота сбора', 'Норма 95%', 'Встречи']

skills_final_columns = ['ВЭД', 'Дата', 'Звонки', 'Обработанные заявки', 'Норма 88%',
                        'Обработка не позднее 48 часов', 'Норма 85%', 'Полнота сбора',
                        'Норма 95%', 'Встречи', 'Звонки (3 мес)',
                        'Звонки / Норма', 'Обработанные заявки (3 мес)',
                        'Норма 88% (3 мес)', 'Обработанные заявки / Норма', 'Обработка не позднее 48 часов (3 мес)',
                        'Норма 85% (3 мес)', '48 часов / Норма', 'Полнота сбора (3 мес)', 'Норма 95% (3 мес)',
                        'Полнота сбора / Норма', 'Встречи (3 мес)', 'Встречи / Норма']


def preprocess_skills(skills: dict) -> pd.DataFrame:
    assert len(skills.keys()) >= 3, 'Переданы данные по навыкам менее чем за квартал'

    # Переносим данные по месячным скилам в один дата-фрейм
    df_list = []
    for key in list(skills.keys())[-3:]:
        month_df = skills[key]
        month_df['Дата'] = key[3:10]
        df_list.append(month_df)
    skills_per_month = pd.concat(df_list, axis=0)
    skills_per_month.columns = skills_columns
    skills_per_month.drop('N', axis=1, inplace=True)
    skills_per_month.reset_index(drop=True, inplace=True)

    ved_list = skills_per_month['ВЭД'].unique()
    df_list = []

    # Проходим по всем ВЭД
    for i, ved in enumerate(ved_list):
        ved_skills = skills_per_month[skills_per_month['ВЭД'] == ved].copy()  # Делаем срез по данному ВЭД
        ved_skills.reset_index(drop=True, inplace=True)

        # Генерим для ВЭД новые признаки
        for column in columns_to_prepare:
            ved_skills[column] = ved_skills[column].astype(str).str.replace(',', '').astype('float')

            # Новый столбец (будут собраны данные за прошедшие 3 мес)
            new_column = column + ' (3 мес)'

            # По каждой колонке суммируем результаты за 3 мес
            ved_skills[new_column] = ved_skills[column].rolling(3, min_periods=1).sum()
        ved_skills_per_quarter = ved_skills.iloc[-1].copy()

        # Считаем отношения между результатами за 3 мес и нормами
        ved_skills_per_quarter['Звонки / Норма'] = ved_skills_per_quarter['Звонки (3 мес)'] / ved_skills_per_quarter[
            'Звонки норма (3 мес)']
        ved_skills_per_quarter['Обработанные заявки / Норма'] = ved_skills_per_quarter['Обработанные заявки (3 мес)'] / \
                                                                ved_skills_per_quarter['Норма 88% (3 мес)']
        ved_skills_per_quarter['48 часов / Норма'] = ved_skills_per_quarter['Обработка не позднее 48 часов (3 мес)'] / \
                                                     ved_skills_per_quarter['Норма 85% (3 мес)']
        ved_skills_per_quarter['Полнота сбора / Норма'] = ved_skills_per_quarter['Полнота сбора (3 мес)'] / \
                                                          ved_skills_per_quarter['Норма 95% (3 мес)']
        ved_skills_per_quarter['Встречи / Норма'] = ved_skills_per_quarter['Встречи (3 мес)'] / ved_skills_per_quarter[
            'Встречи норма (3 мес)']
        ved_skills_per_quarter.fillna(0.0, inplace=True)  # Заполняем NaN там, где возникло деление на 0
        df_list.append(ved_skills_per_quarter)

    # Формируем дата-фрейм по всем сотрудникам
    skills_per_quarter = pd.DataFrame(df_list)

    # Удаляем ВЭД с нулями по всем скилам
    skills_per_quarter = skills_per_quarter[skills_per_quarter[columns_to_check].sum(axis=1) != 0]

    skills_per_quarter.reset_index(drop=True, inplace=True)
    skills_per_quarter.drop(['Звонки норма', 'Встречи норма', 'Звонки норма (3 мес)', 'Встречи норма (3 мес)'], axis=1,
                            inplace=True)
    skills_per_quarter = skills_per_quarter.reindex(columns=skills_final_columns)

    return skills_per_quarter


# 4. Concatenate pd.concat([all_bp, time_process, skills])


def to_ohe(feature, d):
    vector = [0] * len(d)
    if feature in d:
        vector[d[feature]] = 1
    else:
        vector[d['UNK']] = 1
    return vector


def to_bow(feature, d):
    vector = [0] * len(d)
    values = feature.split('__')
    for value in values:
        if value in d:
            vector[d[value]] = 1
        else:
            vector[d['UNK']] = 1
    return vector


def prepare_cat_features(data):
    x_cat = []
    for row in np.array(data):
        prepared_row = []
        prepared_row += to_ohe(row[2], partner_dict)
        prepared_row += to_ohe(row[3], manager_dict)
        prepared_row += to_bow(row[4], position_dict)
        x_cat.append(prepared_row)
    x_cat = np.array(x_cat, dtype=np.float)
    return x_cat


def preprocess_data(bp_data, skills_data, time_data):
    bp_data_unique = preprocess_bp(bp_data.copy())
    bp_id_list = bp_data_unique['Бизнес процесс'].to_list()
    bp_cat_features = prepare_cat_features(bp_data_unique)

    skills_per_quarter = preprocess_skills(skills_data.copy())
    median_time = preprocess_time(time_data.copy())

    skills_time = skills_per_quarter.merge(median_time, on=["ВЭД", "Дата"], how='inner')
    ved_list = skills_time['ВЭД'].to_list()

    num_features = scaler.transform(skills_time.iloc[:, 2:])

    prepared_data = []
    for bp in bp_cat_features:
        prepared_data.append(np.hstack((num_features, bp.reshape((1,-1)).repeat(skills_time.shape[0], axis=0))))
    prepared_data = np.asarray(prepared_data)

    return prepared_data, bp_id_list, ved_list

# 6. Making Json into variable --> results TODO


if __name__ == '__main__':
    test_data_path = os.path.join(data_path, 'ved_test.xlsx')
    skills_path = os.path.join(data_path, 'ved_bp_skills_3.xlsx')
    time_path = os.path.join(data_path, 'ved_bp_processing_time.csv')

    test_data = pd.read_excel(test_data_path, sheet_name=None, header=0)
    skills = pd.read_excel(skills_path, sheet_name=None, header=[1])
    time = pd.read_csv(time_path)

    test_bp = test_data['БП ']
    test_bp.drop('Ответственный', inplace=True, axis=1)
    orders = ['Бизнес-процесс 00-058355 от 09.12.2021 14:28:22',
              'Бизнес-процесс 00-059676 от 03.01.2022 10:28:19']
    test_examples = []
    for order in orders:
        test_examples.append(test_bp[test_bp['Бизнес процесс'] == order])

    test_examples = pd.concat(test_examples)
    test_examples.reset_index(drop=True, inplace=True)

    prepared_data, bp_id_list, ved_list = preprocess_data(test_examples.copy(),
                                                          skills.copy(),
                                                          time.copy())

    print(prepared_data.shape)
    print(bp_id_list)
    print(ved_list)
