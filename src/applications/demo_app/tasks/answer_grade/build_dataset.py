import  pandas
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utilities.transform_data import *
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'


def build_train_dataset_and_vocab_pipeline(config):
    batch_size = config['training']['batch_size']
    used_model = config['net_structure']['model']
    train_map_str = config['train_text_transforming_adaptor'][used_model]
    train_text_transforming_adaptor = []
    for map_str in train_map_str.values():
        train_text_transforming_adaptor.append(eval(map_str))
    valid_map_str = config['valid_text_transforming_adaptor'][used_model]
    valid_text_transforming_adaptor = []
    for map_str in valid_map_str.values():
        valid_text_transforming_adaptor.append(eval(map_str))
    # step 1: load the raw dataset
    print("loading train and valid corpus...")
    train_set, valid_set = load_train_valid_corpus(config)
    # step 2: build the tokenizers
    print("building the tokenizer...")
    build_tokenizer_into_config(config, train_text_transforming_adaptor)
    # step 3: build the vocab
    print("building the vocab...")
    build_vocab_into_config(config, train_text_transforming_adaptor, train_set)
    # step 4: build the vectors
    print("building the vectors...")
    build_vectors_from_pretrained_into_config(config, train_text_transforming_adaptor)
    # step 5: build the special tokens
    print("building the special tokens...")
    build_special_tokens_into_config(config, train_text_transforming_adaptor)
    # step 6: update the model config
    update_some_config(config)
    # step 7: create the map-style Dataset
    print("building the train and valid Dataset...")
    train_set = DatasetGenerator(train_set, triplet=True)
    valid_set = DatasetGenerator(valid_set)
    config['net_structure']['dataset'].update({'train_set_size': len(train_set)})
    # step 8: build iterator loader
    # step 8.1 定义校正函数
    train_collate = partial(collate_fn, config, train_text_transforming_adaptor)      # 用偏函数解决参数传递问题。
    valid_collate = partial(collate_fn, config, valid_text_transforming_adaptor)
    # step 8.2 定义iterator loader
    print("building the train and valid iterator...")
    train_iter = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=train_collate)
    valid_iter = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=valid_collate)
    return train_iter, valid_iter


def build_test_dataset_pipeline(config):
    batch_size = 1  # config['testing']['batch_size']
    used_model = config['model_config']['model_name']
    test_map_str = config['test_text_transforming_adaptor'][used_model]
    test_text_transforming_adaptor = []
    for map_str in test_map_str.values():
        test_text_transforming_adaptor.append(eval(map_str))
    # step 1: load the raw dataset
    print("loading the test corpus...")
    test_set = load_test_corpus(config)
    # step 2: build the tokenizers
    print("building the tokenizer...")
    build_tokenizer_into_config(config, test_text_transforming_adaptor)
    # step 7: create the map-style Dataset
    print("building the test Dataset...")
    test_set = DatasetGenerator(test_set)
    # step 8: build iterator loader
    # step 8.1 定义校正函数
    test_collate = partial(collate_fn, config, test_text_transforming_adaptor)  # 用偏函数解决参数传递问题。
    # step 8.2 定义iterator loader
    print("building the test iterator...")
    test_iter = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True,
                            collate_fn=test_collate)
    return test_iter


def load_train_valid_corpus(config):
    # return train set, valid set
    dataset_root = config['dataset_root']
    train_q_file = config['net_structure']['dataset']['train_q_file']
    train_a_file = config['net_structure']['dataset']['train_a_file']
    valid_size = config['net_structure']['dataset']['valid_size']
    random_state = config['general']['random_state']
    data_text_q = get_txt_from_file(dataset_root + os.path.sep + train_q_file)
    data_text_a = get_txt_from_file(dataset_root + os.path.sep + train_a_file)
    data_set = list(zip(data_text_q, data_text_a))
    train_dataset, valid_dataset = train_test_split(data_set, test_size=valid_size, shuffle=True, random_state=random_state)
    return train_dataset, valid_dataset


def load_test_corpus(config):
    # return test set
    dataset_root = config['dataset_root']
    test_q_file = config['net_structure']['dataset']['test_q_file']
    test_a_file = config['net_structure']['dataset']['test_a_file']
    data_text_q = get_txt_from_file(dataset_root + os.path.sep + test_q_file)
    data_text_a = get_txt_from_file(dataset_root + os.path.sep + test_a_file)
    data_dataset = list(zip(data_text_q, data_text_a))
    return data_dataset


class DatasetGenerator(Dataset):
    def __init__(self, raw_dataset, triplet=False):
        self.raw_dataset = raw_dataset
        self.triplet = triplet
        self.data_len = len(self.raw_dataset)

    def __getitem__(self, index):
        if not self.triplet:
            return self.raw_dataset[index][0], self.raw_dataset[index][1]
        else:
            i = random.randint(0, self.data_len - 1)
            while self.raw_dataset[i][0] == self.raw_dataset[index][0]:
                i = random.randint(0, self.data_len - 1)
            return self.raw_dataset[index][0], self.raw_dataset[index][1], self.raw_dataset[i][1]

    def __len__(self):
        return self.data_len


def update_some_config(config):
    used_model = config['net_structure']['model']
    if ('model_config' not in config.keys()) or (config['model_config'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'model_config': {}})
    config['model_config'].update({'model_name': used_model})
    config['model_config'].update(config['model'][used_model])    # 把配置文件中模型的配置参数复制到model_config项中。

