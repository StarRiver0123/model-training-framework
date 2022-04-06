import numpy as np
import torch.nn.functional as F
from src.utilities.load_data import *
from src.modules.trainer.trainer_framework import Trainer
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import textbrewer
from functools import partial
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
from build_dataset import *
from build_model import *
# from src.applications.demo_app.tasks.ner.build_dataset import *
# from src.applications.demo_app.tasks.ner.build_model import NERModel


def simple_adaptor(batch, model_outputs):
    return {'logits': model_outputs.logits,
            'hidden': model_outputs.hidden_states,
            'labels': batch[1]}


def epoch_validate(model, step, evaluator, valid_iter, max_len=512, device='cuda', batch_interval_for_log=10):
    model.eval()
    with torch.no_grad():
        total_evaluation = 0
        mean_evaluation = 0
        for i, valid_example in enumerate(valid_iter):
            if i % batch_interval_for_log == 0:
                do_log = True
            else:
                do_log = False
            if valid_example[0].size(1) > max_len:
                source = valid_example[0][:, :max_len].to(device)
            else:
                source = valid_example[0].to(device)
            target = valid_example[1].to('cpu')
            logits = model(source).logits
            predict = F.softmax(logits, dim=-1).argmax(dim=-1).to('cpu')
            batch_evaluation = evaluator(predict, target)
            # 模型的输出维度是N，L，D_target_vocab_len
            total_evaluation += batch_evaluation
            mean_evaluation = total_evaluation / (i+1)
            if do_log:
                print("validating batch: %3d; batch evaluation: %0.5f, mean evaluation: %0.5f"%(i, batch_evaluation, mean_evaluation))
                print("Source code:    " + ' '.join(str(index.item()) for index in source[0, :]))
                print("Target code:    " + str(target[0].item()))
                print("Predict code:   " + str(predict[0].item()) + '\n')
    model.train()


def distil_model(config):
    epochs = config['training']['epochs']
    device = config['general']['device']
    check_point_root = config['check_point_root']
    log_dir = config['log_root'] + os.path.sep + 'tblogs'
    s_model = config['net_structure']['student_model']
    max_len = config['model'][s_model]['max_len']
    batch_interval_for_log = config['training']['batch_interval_for_log']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # step 1: build dataset and vocab
    train_iter, valid_iter = build_train_dataset_and_vocab_pipeline(config)
    # step 2: build model
    model = TrainingDistilledStudentModel(config)
    # step 3: get the training config and distilling config
    training_config = TrainingConfig(output_dir=check_point_root, log_dir=log_dir, device=device)
    distilling_config = DistillationConfig(
        temperature=2,
        hard_label_weight=0,
        kd_loss_type='ce',
        probability_shift=False,
        intermediate_matches=[
        {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
        {'layer_T': 5, 'layer_S': 1, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
        {'layer_T': 11, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1}
    ])
    # step 4: build distiller
    distiller = GeneralDistiller(train_config=training_config, distill_config=distilling_config,
                                 model_T=model.teacher_model, model_S=model.model,
                                 adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)
    callback_fn = partial(epoch_validate, evaluator=model.evaluator, valid_iter=valid_iter, max_len=max_len,
                          device=device, batch_interval_for_log=batch_interval_for_log)
    # step 5: start distilling
    with distiller:
        distiller.train(optimizer=model.optimizer, dataloader=train_iter, num_epochs=epochs,
                        scheduler_class=model.lr_scheduler_class, scheduler_args=model.lr_scheduler_args,
                        callback=callback_fn)


