import sys
import torch
import torch.nn.functional as F
from src.utilities.load_data import *
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.models.tester.tester_framework import Tester
from src.tasks.answer_grade.test_model import manage_model_state


def apply_model(arguments):
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    module_obj = sys.modules['src.utilities.load_data']
    use_bert = arguments['tasks'][running_task]['use_bert']
    # max_len = arguments['model'][used_model]['max_len']
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_vocab, _, _, _ = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    text_vocab_stoi = model_vocab['text_vocab_stoi']
    text_vocab_itos = model_vocab['text_vocab_itos']
    if not use_bert:
        tokenizer_name = arguments['tasks'][running_task]['tokenizer']['tokenizer_zh']
        tokenizer = getattr(module_obj, tokenizer_name)
    else:
        tokenizer = get_bert_tokenizer(arguments, language='zh').encode
    while 1:
        input_q = input("请输入问题(输入字符'q'退出)：")
        if input_q == 'q':
            break
        input_a = input("请输入答案(输入字符'q'退出)：")
        if input_a == 'q':
            break
        print('\n')
        if not use_bert:
            input_q = [text_vocab_stoi[word] for word in tokenizer(input_q)]
            input_a = [text_vocab_stoi[word] for word in tokenizer(input_a)]
        else:
            input_q = tokenizer(input_q)
            input_a = tokenizer(input_a)
        input_q = torch.tensor(input_q).unsqueeze(0)         # 模型的输入需要2维。
        input_a = torch.tensor(input_a).unsqueeze(0)  # 模型的输入需要2维。
        input_seq = (input_q, input_a)
        # start to run
        tester.apply(model=model, input_seq=input_seq,
                      compute_predict_func=compute_predict,
                      compute_predict_outer_params={'text_vocab_itos': text_vocab_itos})


# this function needs to be defined from the view of concrete task
def compute_predict(model, input_seq, device, log_string_list, text_vocab_itos):
    # model, data_example, device, log_string_list are from inner tester framework
    # output: predict: 1,L,D
    source = input_seq[0].to(device)
    target = input_seq[1].to(device)
    source_vector, target_vector = model.model(source, target)
    evaluation = model.evaluator(source_vector, target_vector)
    # evaluation2 = model.evaluator(source.float(), target.float())
    log_string_list.append("Source string:  " + ' '.join(text_vocab_itos[index] for index in source[0]))
    log_string_list.append("Target string:  " + ' '.join(text_vocab_itos[index] for index in target[0]))
    log_string_list.append(f"相似度评分:  {evaluation: 0.3f}")
    return evaluation

