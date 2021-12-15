import torch
import os, sys



# this is a testing framework
# the callback function do_test and do_test need to be implemented in task test codes.
class Tester():
    def __init__(self, arguments):
        self.arguments = arguments
        self.running_task = arguments['general']['running_task']
        self.used_model = arguments['tasks'][self.running_task]['model']
        self.max_len = arguments['model'][self.used_model]['max_len']
        self.batch_interval_for_log = arguments['testing'][self.running_task]['batch_interval_for_log']
        self.batch_size_for_test = arguments['testing'][self.running_task]['batch_size']
        assert self.batch_size_for_test == 1
        self.model_save_mode = arguments['training'][self.running_task]['model_save']['save_mode']   # 读取训练配置
        self.model_save_root = arguments['file']['data']['model']
        self.saved_model_file = arguments['testing'][self.running_task]['saved_model_file']
        self.project_root = arguments['general']['project_root']
        self.device = arguments['general']['device']
        self.pad_token = arguments['dataset']['general']['pad_token']
        self.logger = arguments['general']['logger']

    def test(self, model, test_iter, compute_predict_evaluation_func, compute_predict_evaluation_outer_params):
        model.model.eval()
        with torch.no_grad():
            total_evaluation = 0
            mean_evaluation = 0
            for i, test_example in enumerate(test_iter):
                if len(test_example) < self.batch_size_for_test:
                    continue
                if i % self.batch_interval_for_log == 0:
                    do_log = True
                else:
                    do_log = False
                log_string_list = []
                predict, target, batch_evaluation = compute_predict_evaluation_func(model, test_example, self.max_len, self.device,
                                                       do_log, log_string_list, **compute_predict_evaluation_outer_params)
                # 模型的输出维度是N，L，D_target_vocab_len
                total_evaluation += batch_evaluation
                mean_evaluation = total_evaluation / (i+1)
                if do_log:
                    self.logger.info("testing sentence: %d; sentence evaluation: %0.5f, mean evaluation: %0.5f",
                                     i, batch_evaluation, mean_evaluation,
                                     extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                    for log_str in log_string_list:
                        self.logger.info(log_str, extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
                    log_string_list.clear()
        print("testing finished.")


    def apply(self, model, input_seq, compute_predict_func, compute_predict_outer_params):
        model.model.eval()
        with torch.no_grad():
            log_string_list = []
            predict = compute_predict_func(model, input_seq, self.device, log_string_list, **compute_predict_outer_params)
            for log_str in log_string_list:
                self.logger.info(log_str, extra={'file_name': os.path.basename(__file__), 'line_no': sys._getframe().f_lineno})
            log_string_list.clear()


    def load_model(self, get_model_state_func, get_model_state_outer_params):
        model_file_path = self.project_root + os.path.sep + self.model_save_root + os.path.sep + self.running_task + os.path.sep + self.saved_model_file
        if self.model_save_mode == 'model':
            model = torch.load(model_file_path)
            return model, None, None, None
        elif self.model_save_mode == 'state':
            loaded_weights = torch.load(model_file_path)
            model_states = get_model_state_func(self.arguments, loaded_weights, **get_model_state_outer_params)
            return model_states