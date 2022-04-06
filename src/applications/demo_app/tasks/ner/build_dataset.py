import  pandas
from torch.utils.data import Dataset, DataLoader
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
    train_set = DatasetGenerator(train_set)
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
    train_set_file = config['net_structure']['dataset']['train_set_file']
    valid_set_file = config['net_structure']['dataset']['valid_set_file']
    train_list = get_txt_from_file(dataset_root + os.path.sep + train_set_file)
    valid_list = get_txt_from_file(dataset_root + os.path.sep + valid_set_file)
    train_set = []
    for i, text in enumerate(train_list):
        if i % 2 == 0:
            s = text.strip()#.split()
        else:
            t = text.strip()#.split()
            train_set.append([s, t])
    valid_set = []
    for i, text in enumerate(valid_list):
        if i % 2 == 0:
            s = text.strip()#.split()
        else:
            t = text.strip()#.split()
            valid_set.append([s, t])
    return train_set, valid_set


def load_test_corpus(config):
    # return test set
    dataset_root = config['dataset_root']
    test_set_file = config['net_structure']['dataset']['test_set_file']
    test_list = get_txt_from_file(dataset_root + os.path.sep + test_set_file)
    test_set = []
    for i, text in enumerate(test_list):
        if i % 2 == 0:
            s = text.strip()#.split()
        else:
            t = text.strip()#.split()
            test_set.append([s, t])
    return test_set


class DatasetGenerator(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __getitem__(self, index):
        return self.raw_dataset[index][0], self.raw_dataset[index][1]

    def __len__(self):
        return len(self.raw_dataset)


def update_some_config(config):
    used_model = config['net_structure']['model']
    if ('model_config' not in config.keys()) or (config['model_config'] is None):  # 用于保存模型参数和测试部署阶段使用
        config.update({'model_config': {}})
    config['model_config'].update({'model_name': used_model})
    config['model_config'].update(config['model'][used_model])    # 把配置文件中模型的配置参数复制到model_config项中。

