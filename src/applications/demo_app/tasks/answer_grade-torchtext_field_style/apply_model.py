import sys
import torch
import torch.nn.functional as F
from src.utilities.load_data import *
from src.modules.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.modules.tester.tester_framework import Tester
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_model import load_model_states_into_config
from build_model import create_inference_model, create_evaluator
tokenizer_package_path = r'src.modules.tokenizers.tokenizer'

def apply_model(config):
    # step 1: load model and config states
    load_model_states_into_config(config)
    # step 2: create model
    model = create_inference_model(config)
    # step 2': create evaluator if required
    evaluator = create_evaluator(config)
    # step 3: load vocab
    text_vocab_stoi = config['model_vocab']['vocab_stoi']
    text_vocab_itos = config['model_vocab']['vocab_itos']
    # step 4: create tokenizer
    use_bert = config['model_config']['use_bert']
    if not use_bert:
        module_obj = sys.modules[tokenizer_package_path]
        tokenizer_name = config['net_structure']['tokenizer']
        tokenizer = getattr(module_obj, tokenizer_name)
    else:
        bert_model_root = config['bert_model_root']
        bert_model_file = bert_model_root + os.path.sep + config['net_structure']['bert_model']['bert_model_file']
        tokenizer = BertTokenizer.from_pretrained(bert_model_file).encode
    # step 5: start main process
    while 1:
        # step 5-1: get the input
        input_q = input("请输入问题(输入字符'q'退出)：")
        if input_q == 'q':
            break
        input_a = input("请输入答案(输入字符'q'退出)：")
        if input_a == 'q':
            break
        print('\n')
        # step 5-2: get the token ids
        if not use_bert:
            input_q = [text_vocab_stoi[word] for word in tokenizer(input_q)]
            input_a = [text_vocab_stoi[word] for word in tokenizer(input_a)]
        else:
            input_q = tokenizer(input_q)
            input_a = tokenizer(input_a)
        # step 5-3: adjust the input shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        source = torch.tensor(input_q).unsqueeze(0).to(device)         # 模型的输入需要2维。
        target = torch.tensor(input_a).unsqueeze(0).to(device)  # 模型的输入需要2维。
        # step 5-4: start to inference
        model.eval()
        with torch.no_grad():
            source_vector, target_vector = model(source, target)
            similarity = evaluator(source_vector, target_vector)
            print("Source string:  " + ' '.join(text_vocab_itos[index] for index in source[0]))
            print("Target string:  " + ' '.join(text_vocab_itos[index] for index in target[0]))
            print(f"相似度评分:  {similarity: 0.3f}")



# this function needs to be defined from the view of concrete task
# def compute_predict(model, evaluator, input_seq, device, text_vocab_itos):
#     # model, data_example, device, log_string_list are from inner tester framework
#     # output: predict: 1,L,D
#     source = input_seq[0].to(device)
#     target = input_seq[1].to(device)
#     source_vector, target_vector = model.model(source, target)
#     evaluation = evaluator(source_vector, target_vector)
#     print("Source string:  " + ' '.join(text_vocab_itos[index] for index in source[0]))
#     print("Target string:  " + ' '.join(text_vocab_itos[index] for index in target[0]))
#     print(f"相似度评分:  {evaluation: 0.3f}")


