import torch
import os
import time
import sys
import torch.nn as nn
from src.utilities.create_logger import create_logger

# this is a training framework
# the callback function epoch_train and epoch_validate need to be implemented in task train codes.
class Trainer():
    def __init__(self, config):
        self.config = config
        self.used_model = config['net_structure']['model']
        self.used_evaluator = config['net_structure']['evaluator']
        self.max_len = config['model'][self.used_model]['max_len']
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.batch_interval_for_log = config['training']['batch_interval_for_log']
        self.validating = config['training']['validating']
        self.loss_threshold = config['training']['loss_threshold']
        self.evaluation_threshold = config['training']['evaluation_threshold']
        self.model_save_root = config['check_point_root']
        self.device = config['general']['device']
        self.pad_token = config['dataset']['general_symbol']['pad_token']
        self.logger = config['logger']
        self.max_norm = config['training']['max_norm']
        self.resume_from_check_point = config['training']['resume_from_check_point']
        self.model_check_point_file = config['training']['model_check_point_file']
        self.save_check_point = config['training']['save_check_point']
        self.check_point_epoch_step = config['training']['check_point_epoch_step']

    def train(self, model, train_iter, compute_predict_loss_func, compute_predict_loss_outer_params, valid_iter=None,  compute_predict_evaluation_func=None, compute_predict_evaluation_outer_params=None, save_model_state_func=None, save_model_state_outer_params=None):
        best_loss = self.loss_threshold
        best_evaluation = self.evaluation_threshold
        start_epoch = 0
        if self.resume_from_check_point:
            model_file_path = self.model_save_root + os.path.sep + self.model_check_point_file
            check_point = torch.load(model_file_path)
            start_epoch = check_point['epoch'] + 1
            model.model.load_state_dict(check_point['model'])
            check_point['model'] = None
            model.optimizer.load_state_dict(check_point['optimizer'])
            check_point['optimizer'] = None
            model.lr_scheduler.load_state_dict(check_point['lr_scheduler'])
            check_point['lr_scheduler'] = None
            del check_point
        for epoch in range(start_epoch, self.epochs):
            epoch_loss = self._epoch_train(model, epoch, train_iter, compute_predict_loss_func, compute_predict_loss_outer_params)
            if self.save_check_point and (epoch % self.check_point_epoch_step == 0):
                self._save_model(model, is_check_point=True, epoch=epoch, loss=epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.validating:
                    epoch_evaluation = self._epoch_validate(model, epoch, valid_iter, compute_predict_evaluation_func, compute_predict_evaluation_outer_params)
                    if epoch_evaluation > best_evaluation:
                        best_evaluation = epoch_evaluation
                        self._save_model(model, is_check_point=False, epoch=epoch, loss=best_loss, evaluation=best_evaluation, save_model_state_func=save_model_state_func, save_model_state_outer_params=save_model_state_outer_params)
                self._save_model(model, is_check_point=False, epoch=epoch, loss=best_loss, evaluation=best_evaluation, save_model_state_func=save_model_state_func, save_model_state_outer_params=save_model_state_outer_params)
        print("training finished.")

    def _epoch_train(self, model, epoch, train_iter, compute_predict_loss_func, compute_predict_loss_outer_params):
        model.model.train()
        total_loss = 0
        mean_loss = 0
        for i, train_example in enumerate(train_iter):
            if len(train_example) < self.batch_size:      # 相当于pytorch dataloader里使用的drop_last参数
                continue
            if i % self.batch_interval_for_log == 0:
                do_log = True
            else:
                do_log = False
            log_string_list = []
            logit, target, batch_loss = compute_predict_loss_func(model, train_example, self.max_len, self.device, do_log, log_string_list, **compute_predict_loss_outer_params)
            model.optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=self.max_norm)   #梯度裁剪
            model.optimizer.step()
            if model.lr_scheduler is not None:
                model.lr_scheduler.step()
            total_loss += batch_loss.item()
            mean_loss = total_loss / (i+1)
            if do_log:
                self.logger.info("training epoch-batch: %d-%d; batch_size: %d; batch loss: %0.8f, mean loss: %0.8f",
                                 epoch, i, self.batch_size, batch_loss.item(), mean_loss,
                                 extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                self.logger.info("learning rate: %0.8f", model.optimizer.state_dict()['param_groups'][0]['lr'],
                                 extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                for log_str in log_string_list:
                    self.logger.info(log_str, extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                log_string_list.clear()
        return mean_loss

    def _epoch_validate(self, model, epoch, valid_iter, compute_predict_evaluation_func, compute_predict_evaluation_outer_params):
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
                predict, target, batch_evaluation = compute_predict_evaluation_func(model, valid_example, self.max_len, self.device,
                                                       do_log, log_string_list, **compute_predict_evaluation_outer_params)
                # 模型的输出维度是N，L，D_target_vocab_len
                total_evaluation += batch_evaluation
                mean_evaluation = total_evaluation / (i+1)

                if do_log:
                    self.logger.info("validating epoch-batch: %d-%d; batch_size: %d; batch evaluation: %0.5f, mean evaluation: %0.5f",
                                     epoch, i, self.batch_size, batch_evaluation, mean_evaluation,
                                     extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                    for log_str in log_string_list:
                        self.logger.info(log_str, extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                    log_string_list.clear()
        return mean_evaluation

    def _save_model(self, model, is_check_point=False, epoch=None, loss=None, evaluation=None, save_model_state_func=None, save_model_state_outer_params=None):
        # model_save_path = self.project_root + os.path.sep + self.model_save_root + os.path.sep + self.running_task
        if not os.path.exists(self.model_save_root):
            os.makedirs(self.model_save_root)
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
        if is_check_point:
            file_name = self.model_save_root + os.path.sep + 'check_point' + '_epoch_{:03d}_loss_{:0.5f}_'.format(epoch, loss) + time_stamp + '.pt'
            check_point = {'epoch': epoch,
                           'model': model.model.state_dict(),
                           'optimizer': model.optimizer.state_dict(),
                           'lr_scheduler': model.lr_scheduler.state_dict()
                           }
            torch.save(check_point, file_name)
        else:
            file_name = self.model_save_root + os.path.sep + 'model' + '_epoch_{:03d}_loss_{:0.5f}_'.format(epoch, loss) + self.used_evaluator + '_{:0.5f}_'.format(evaluation) + time_stamp + '.pt'
            model_states = save_model_state_func(model, self.config, **save_model_state_outer_params)
            training_states = {'mean_loss': loss, self.used_evaluator: evaluation, 'epoch': epoch}
            model_states.update({'training_states': training_states})
            torch.save(model_states, file_name)
