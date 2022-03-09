import os
import csv
import pandas as pd
import numpy as np

raw_data_root = "../../../../../dataset/for_edu_dialogue_robot/原始数据/"
kemu_list = ['高中历史', '高中地理', '高中生物', '高中政治']

def process_raw_data():
    data_table = [['kemu', 'zhangjie', 'zhishidian', 'timu']]
    for kemu in kemu_list:
        folder = raw_data_root + kemu + '/origin'
        file_list = os.listdir(folder)
        for file in file_list:
            zhangjie = file.rstrip('.csv')
            file_path = folder + '/' + file
            raw_data = pd.read_csv(file_path).item
            for line in raw_data:
                content = line.replace('\n', '')
                if content.find('[知识点：]') == -1:
                    continue
                content = content.split('[知识点：]')
                timu = content[0].replace('[题目]', '').replace('知识点：', '')
                zhishidian_list = content[1].split(',')
                for zhishidian in zhishidian_list:
                    data_table.append([kemu, zhangjie, zhishidian, timu])
    # data = pd.DataFrame(np.array(data_table), columns=['kemu', 'zhangjie', 'zhishidian', 'timu'])
    # data.to_csv('../../../../../dataset/for_edu_dialogue_robot/kemu.csv', index=False)
    with open('../../../../../dataset/for_edu_dialogue_robot/kemu.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data_table)
    return np.array(data_table)


def create_entity(data_table):
    entity_table = np.array([[':ID', 'name', ':LABEL']])
    for i, label in enumerate(data_table[0]):
        names = list(set(data_table[1:, i]))
        ids = ['id_' + str(hash(name)) for name in names]
        labels = [label] * len(names)
        entity_table = np.concatenate((entity_table, (np.array(list(zip(ids, names, labels))))), axis=0)
    # data = pd.DataFrame(np.array(entity_table), columns=[':ID', 'name', ':LABEL'])
    # data.to_csv('../../../../../dataset/for_edu_dialogue_robot/entity.csv', index=False)
    with open('../../../../../dataset/for_edu_dialogue_robot/entity.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(entity_table)


def create_relation(data_table):
    relation_table = np.array([[':START_ID', 'name', ':END_ID', ':TYPE']])
    for i in range(len(data_table[0]) - 1):
        pairs = np.array(list(set(zip(data_table[1:, i], data_table[1:, i+1]))))
        names = types = [data_table[0][i] + '2' + data_table[0][i+1]] * len(pairs)
        start_ids = ['id_' + str(hash(name)) for name in pairs[:, 0]]
        end_ids = ['id_' + str(hash(name)) for name in pairs[:, 1]]
        relation_table = np.concatenate((relation_table, (np.array(list(zip(start_ids, names, end_ids, types))))), axis=0)
    # data = pd.DataFrame(np.array(entity_table), columns=[':START_ID', 'name', ':END_ID', ':TYPE'])
    # data.to_csv('../../../../../dataset/for_edu_dialogue_robot/relation.csv', index=False)
    with open('../../../../../dataset/for_edu_dialogue_robot/relation.csv', 'w', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(relation_table)


if __name__ == '__main__':
    data = process_raw_data()
    print(len(data))

    # create_entity(data)
    create_relation(data)
    print("ok.")

    #
    # # path = 'data_combine/'
    # data = pd.read_csv('../../../../../dataset/for_edu_dialogue_robot/kemu.csv')
    # entity = None
    # for i in data:
    #     names = list(set(eval('data.' + i)))
    #     ids = [hash(j) for j in names]
    #     labels = [i for j in names]
    #     if entity is None:
    #         entity = pd.DataFrame(np.array([ids, names, labels]).transpose())
    #     else:
    #         entity = entity.append(pd.DataFrame(np.array([ids, names, labels]).transpose()))
    #
    # entity.columns = [':ID', 'name', ':LABEL']
    # print('*', entity[':ID'][0], '*')
    # entity[':ID'] = entity[':ID'].map(long_num_str)
    # print('*', entity[':ID'][0], '*')
    # entity.to_csv('../../../../../dataset/for_edu_dialogue_robot/entity_km.csv', index=False)