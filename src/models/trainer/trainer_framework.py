import torch
import os
import time
import sys
from src.utilities.create_logger import create_logger

# this is a training framework
# the callback function epoch_train and epoch_validate need to be implemented in task train codes.
class Trainer():
    def __init__(self, arguments):
        self.arguments = arguments
        self.running_task = arguments['general']['running_task']
        self.used_model = arguments['tasks'][self.running_task]['model']
        self.epochs = arguments['training'][self.running_task]['epochs']
        self.batch_size = arguments['training'][self.running_task]['batch_size']
        self.batch_interval_for_log = arguments['training'][self.running_task]['batch_interval_for_log']
        self.validating = arguments['training'][self.running_task]['validating']
        self.loss_threshold = arguments['training'][self.running_task]['loss_threshold']
        self.evaluation_threshold = arguments['training'][self.running_task]['evaluation_threshold']
        self.save_model = arguments['training'][self.running_task]['model_save']['save_model']
        self.model_save_mode = arguments['training'][self.running_task]['model_save']['save_mode']
        self.model_save_root = arguments['file']['data']['model']
        self.project_root = arguments['general']['project_root']
        self.device = arguments['general']['device']
        self.pad_token = arguments['dataset']['general']['pad_token']
        self.logger = arguments['general']['logger']

    def train(self, model, train_iter, compute_predict_func, compute_predict_outer_params, valid_iter=None, get_model_state_func=None, get_model_state_outer_params=None):
        best_loss = self.loss_threshold
        best_evaluation = self.evaluation_threshold
        for epoch in range(self.epochs):
            epoch_loss = self._epoch_train(model, train_iter, compute_predict_func, compute_predict_outer_params, epoch)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.validating:
                    epoch_evaluation = self._epoch_validate(model, valid_iter, compute_predict_func, compute_predict_outer_params, epoch)
                    if epoch_evaluation > best_evaluation:
                        best_evaluation = epoch_evaluation
                        if self.save_model:
                            self._save_model(model, best_loss, best_evaluation, epoch, get_model_state_func, get_model_state_outer_params)
                elif self.save_model:
                    self._save_model(model, best_loss, best_evaluation, epoch, get_model_state_func, get_model_state_outer_params)
        print("training finished.")

    def _epoch_train(self, model, train_iter, compute_predict_func, compute_predict_outer_params, epoch):
        model.model.train()
        total_loss = 0
        mean_loss = 0
        for i, train_example in enumerate(train_iter):
            if len(train_example) < self.batch_size:
                continue
            if i % self.batch_interval_for_log == 0:
                do_log = True
            else:
                do_log = False
            log_string_list = []
            logit, target = compute_predict_func(model, train_example, self.device, do_log, log_string_list, **compute_predict_outer_params)
            # 模型的输出维度是N，L，D_target_vocab_len,
            # pytorch CrossEntropyLoss的输入维度有两种方式：
            # （1） input为N，C；target为N，需要对predict做reshape（-1，D_target_vocab_len）
            # （2） input为N，C，L，target为N，L。要把分类放在第二维，需要对predict进行转置transpose(-1,-2)
            logit_flatten = logit.reshape(-1, logit.size(-1))
            target_flatten = target.reshape(-1)
            batch_loss = model.criterion(logit_flatten, target_flatten)
            model.optimizer.zero_grad()
            batch_loss.backward()
            model.optimizer.step()
            if model.lr_scheduler is not None:
                model.lr_scheduler.step()
            total_loss += batch_loss.item()
            mean_loss = total_loss / (i+1)
            if do_log:
                self.logger.info("training epoch-batch: %d-%d; batch_size: %d; batch loss: %0.3f, mean loss: %0.3f",
                                 epoch, i, self.batch_size, batch_loss.item(), mean_loss,
                                 extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                self.logger.info("learning rate: %0.6f", model.optimizer.state_dict()['param_groups'][0]['lr'],
                                 extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                for log_str in log_string_list:
                    self.logger.info(log_str, extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                log_string_list.clear()
        return mean_loss

    def _epoch_validate(self, model, valid_iter, compute_predict_func, compute_predict_outer_params, epoch):
        model.model.eval()
        with torch.no_grad():
            total_evaluation = 0
            mean_evaluation = 0
            for i, valid_example in enumerate(valid_iter):
                if len(valid_example) < self.batch_size:
                    continue
                if i % self.batch_interval_for_log == 0:
                    do_log = True
                else:
                    do_log = False
                log_string_list = []
                predict, target = compute_predict_func(model, valid_example, self.device,
                                                       do_log, log_string_list, **compute_predict_outer_params)
                # 模型的输出维度是N，L，D_target_vocab_len
                batch_evaluation = model.evaluator(predict, target)
                total_evaluation += batch_evaluation
                mean_evaluation = total_evaluation / (i+1)

                if do_log:
                    self.logger.info("validating epoch-batch: %d-%d; batch_size: %d; batch evaluation: %0.3f, mean evaluation: %0.3f",
                                     epoch, i, self.batch_size, batch_evaluation, mean_evaluation,
                                     extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                    for log_str in log_string_list:
                        self.logger.info(log_str, extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                    log_string_list.clear()
        return mean_evaluation

    def _save_model(self, model, loss, evaluation, epoch, get_model_state_func, get_model_state_outer_params):
        model_save_path = self.project_root + os.path.sep + self.model_save_root + os.path.sep + self.running_task
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        if self.model_save_mode == 'state':
            file_name = model_save_path + os.path.sep + time_stamp + '_{:03d}_{:0.3f}_{:0.3f}'.format(epoch, loss, evaluation) + '_state_dict.pmd'
            model_state = get_model_state_func(model, self.arguments, **get_model_state_outer_params)
            training_states = {'mean_loss': loss,
                                'mean_evaluation': evaluation,
                                'epoch': epoch}
            model_state.update({'training_states': training_states})
            torch.save(model_state, file_name)
        elif self.model_save_mode == 'model':
            file_name = model_save_path + os.path.sep + time_stamp + '_{:03d}-{:0.3f}-{:0.3f}'.format(epoch, loss,
                                                                                              evaluation) + '_model.pmd'
            torch.save(model.model, file_name)