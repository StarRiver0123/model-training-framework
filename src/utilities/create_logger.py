import logging
import time
import os
import sys

def create_logger(arguments):
    running_task = arguments['general']['running_task']
    project_root = arguments['general']['project_root']
    log_root = arguments['file']['data']['log']
    sys_log_level = arguments['logging']['general']['sys_log_level']
    file_log_level = arguments['logging']['general']['file_log_level']
    console_log_level = arguments['logging']['general']['console_log_level']

    if sys_log_level.lower() == 'debug':
        sys_log_level = logging.DEBUG
    elif sys_log_level.lower() == 'info':
        sys_log_level = logging.INFO
    elif sys_log_level.lower() == 'warning':
        sys_log_level = logging.WARNING
    elif sys_log_level.lower() == 'error':
        sys_log_level = logging.ERROR
    elif sys_log_level.lower() == 'critical':
        sys_log_level = logging.CRITICAL
    else:
        sys_log_level = logging.INFO

    if file_log_level.lower() == 'debug':
        file_log_level = logging.DEBUG
    elif file_log_level.lower() == 'info':
        file_log_level = logging.INFO
    elif file_log_level.lower() == 'warning':
        file_log_level = logging.WARNING
    elif file_log_level.lower() == 'error':
        file_log_level = logging.ERROR
    elif file_log_level.lower() == 'critical':
        file_log_level = logging.CRITICAL
    else:
        file_log_level = logging.INFO

    if console_log_level.lower() == 'debug':
        console_log_level = logging.DEBUG
    elif console_log_level.lower() == 'info':
        console_log_level = logging.INFO
    elif console_log_level.lower() == 'warning':
        console_log_level = logging.WARNING
    elif console_log_level.lower() == 'error':
        console_log_level = logging.ERROR
    elif console_log_level.lower() == 'critical':
        console_log_level = logging.CRITICAL
    else:
        console_log_level = logging.INFO

    # 下面的四行是为了进行性能优化。
    logging._srcfile = None
    logging.logThreads = False
    logging.logMultiprocessing = False
    logging.logProcesses = False

    logger = logging.getLogger(__name__)
    logger.setLevel(sys_log_level)
    log_format = logging.Formatter("%(asctime)s - %(file_name)s[line:%(line_no)s] - %(levelname)s: %(message)s")

    log_path = project_root + os.path.sep + log_root + os.path.sep + running_task
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    log_file = log_path + os.path.sep + time_stamp + '.log'
    fh = logging.FileHandler(log_file, encoding='utf-8')  # 如果不指定编码，可能会出现Python UnicodeEncodeError: 'gbk' codec can't encode character
    fh.setFormatter(log_format)
    fh.setLevel(file_log_level)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(log_format)
    sh.setLevel(console_log_level)
    logger.addHandler(sh)

    return logger

