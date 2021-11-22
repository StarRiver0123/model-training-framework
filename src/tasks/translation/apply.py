import sys
import torch
import torch.nn.functional as F
from src.models.models.base_component import gen_pad_only_mask, gen_seq_only_mask
from src.models.tester.tester_framework import Tester
from src.tasks.translation.test_model import manage_model_state
from src.utilities.load_data import *

def apply_model(arguments):
    # get the tester
    tester = Tester(arguments)
    # get the model
    model, model_creation_args, extra_states, training_states = tester.load_model(get_model_state_func=manage_model_state, get_model_state_outer_params={})
    running_task = arguments['general']['running_task']
    used_model = arguments['tasks'][running_task]['model']
    src_vocab_stoi = arguments['model'][used_model]['src_vocab_stoi']
    src_vocab_itos = arguments['model'][used_model]['src_vocab_itos']
    tgt_vocab_stoi = arguments['model'][used_model]['tgt_vocab_stoi']
    tgt_vocab_itos = arguments['model'][used_model]['tgt_vocab_itos']
    max_len = arguments['model'][used_model]['max_len']

    module_obj = sys.modules['src.utilities.load_data']
    trans_direct = arguments['tasks'][running_task]['trans_direct']
    use_bert = arguments['tasks'][running_task]['word_vector']['use_bert']
    if (use_bert != 'static') and (use_bert != 'dynamic'):
        start_token = arguments['dataset']['general']['start_token']
        end_token = arguments['dataset']['general']['end_token']
        if trans_direct == 'en2zh':
            tokenizer_name = arguments['tasks'][running_task]['word_vector']['tokenizer_en']
        elif trans_direct == 'zh2en':
            tokenizer_name = arguments['tasks'][running_task]['word_vector']['tokenizer_zh']
        else:
            print("翻译语言不支持")
        tokenizer = getattr(module_obj, tokenizer_name)
    else:
        start_token = arguments['dataset']['bert']['start_token']
        end_token = arguments['dataset']['bert']['end_token']
        if trans_direct == 'en2zh':
            tokenizer = get_bert_tokenizer(arguments, language='en').tokenize
        elif trans_direct == 'zh2en':
            tokenizer = get_bert_tokenizer(arguments, language='zh').tokenize
        else:
            print("翻译语言不支持")
    while 1:
        input_sentence = input("请输入一句英语(input a single 'q' to quit)：")
        if input_sentence == 'q':
            break
        print('\n')
        seqs = [src_vocab_stoi[start_token]] + [src_vocab_stoi[word] for word in tokenizer(input_sentence)] + [src_vocab_stoi[end_token]]
        input_seq = torch.tensor(seqs).unsqueeze(0)
        # start to run
        tester.apply(model=model, input_seq=input_seq,
                      compute_predict_func=compute_predict,
                      compute_predict_outer_params={'src_vocab_itos': src_vocab_itos, 'tgt_vocab_itos': tgt_vocab_itos, 'max_len': max_len})


# this function needs to be defined from the view of concrete task
def compute_predict(model, input_seq, device, log_string_list, src_vocab_itos, tgt_vocab_itos, max_len):
    # model, data_example, device, log_string_list are from inner tester framework
    # output: predict: 1,L,D
    source = input_seq.to(device)
    end_index = source[:, -1]
    # enc_input = model.model.encoder_embedding(source)
    enc_out = model.model.encoder(source)
    target_input = source[:, 0:1]   # initialized as start_token
    for i in range(max_len - 1):
        tgt_mask = gen_seq_only_mask(target_input, target_input)  # L,L
        # dec_input = model.model.decoder_embedding(target_input)
        dec_out = model.model.decoder(target_input, enc_out, tgt_mask=tgt_mask)
        predict = model.model.predictor(dec_out)
        last_word = F.softmax(predict[:, -1], dim=-1).argmax(dim=-1, keepdim=True)     # last_word size: (1,1)
        if last_word.item() == end_index.item():
            break
        target_input = torch.cat([target_input, last_word], dim=-1)
    log_string_list.append(
        "识别出原文: " + ' '.join(src_vocab_itos[index] for index in source[0, 1:-1]))
    log_string_list.append("翻译出译文: " + ' '.join(
        tgt_vocab_itos[index] for index in F.softmax(predict[0, :, :-1], dim=-1).argmax(dim=-1)) + '\n`')
    return predict[:, :-1]

