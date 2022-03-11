import os
import torch
import src
from src.utilities.create_logger import create_logger
from src.utilities.load_data import get_config


# will be called by run.py in the project root, to load all config.
def load_config(config_root_file, config_deploy_file=None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(src.__file__))).strip('/')
    config_project_file = project_root + os.path.sep + config_root_file
    config = get_config(config_project_file)
    app_root = project_root + os.path.sep + config['app_root'].strip('/') + os.path.sep + config['running_app']
    task_root = app_root + os.path.sep + config['task_root'].strip('/') + os.path.sep + config['running_task']
    dataset_root = project_root + os.path.sep + config['dataset_root'].strip('/')
    bert_model_root = project_root + os.path.sep + config['bert_model_root'].strip('/')
    word_vector_root = project_root + os.path.sep + config['word_vector_root'].strip('/')
    log_root = project_root + os.path.sep + config['log_root'].strip('/') + os.path.sep + config['running_app'] + os.path.sep + config['running_task']
    check_point_root = project_root + os.path.sep + config['check_point_root'].strip('/') + os.path.sep + config['running_app'] + os.path.sep + config['running_task']
    running_step = config['running_step']
    if running_step in ['preprocess', 'train_model']:
        config_task_file = task_root + os.path.sep + config['training_config_file']
    elif running_step in ['test_model', 'apply_model']:
        config_task_file = task_root + os.path.sep + config['testing_config_file']
    args = get_config(config_task_file)
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if not os.path.exists(check_point_root):
        os.makedirs(check_point_root)
    logger = create_logger(log_root, args['logging']['sys_log_level'], args['logging']['file_log_level'], args['logging']['console_log_level'])
    if ('general' not in args.keys()) or (args['general'] is None):
        args.update({'general': {}})
    if ('device' not in args['general'].keys()) or (args['general']['device'] not in ['cpu', 'cuda']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args['general'].update({'device': device})
    config.update(args)
    config.update({'project_root': project_root,
                      'app_root': app_root,
                      'task_root': task_root,
                      'dataset_root': dataset_root,
                      'bert_model_root': bert_model_root,
                      'word_vector_root': word_vector_root,
                      'log_root': log_root,
                      'check_point_root': check_point_root,
                      'logger': logger})
    if config_deploy_file is not None:
        create_deploy_folders(config_deploy_file, config['running_app'])
    return config


def create_deploy_folders(config_root_file, running_app):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(src.__file__))).strip('/')
    config_deploy_file = project_root + os.path.sep + config_root_file
    config = get_config(config_deploy_file)
    deploy_root = project_root + os.path.sep + config['deploy_root'].strip('/')
    service_src_root = deploy_root + os.path.sep + running_app + os.path.sep + config['service_src_root'].strip('/')
    service_data_root = deploy_root + os.path.sep + running_app + os.path.sep + config['service_data_root'].strip('/')
    django_server_root = deploy_root + os.path.sep + running_app + os.path.sep + config['django_server_root'].strip('/')
    flask_server_root = deploy_root + os.path.sep + running_app + os.path.sep + config['flask_server_root'].strip('/')
    app_config_file = deploy_root + os.path.sep + running_app + os.path.sep + config['app_config_file'].strip('/')
    flask_start_file = flask_server_root + os.path.sep + config['server_start_file']
    django_start_file = django_server_root + os.path.sep + config['server_start_file']
    if not os.path.exists(deploy_root):
        os.makedirs(deploy_root)
    if not os.path.exists(service_src_root):
        os.makedirs(service_src_root)
    if not os.path.exists(service_data_root):
        os.makedirs(service_data_root)
    if not os.path.exists(django_server_root):
        os.makedirs(django_server_root)
    if not os.path.exists(flask_server_root):
        os.makedirs(flask_server_root)
    if not os.path.exists(flask_start_file):
        os.system(r"echo # coding=utf-8 > {}".format(flask_start_file))
    if not os.path.exists(django_start_file):
        os.system(r"echo # coding=utf-8 > {}".format(django_start_file))
    if not os.path.exists(app_config_file):
        os.system(r"echo # coding=utf-8 > {}".format(app_config_file))
