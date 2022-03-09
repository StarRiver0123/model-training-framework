import os, sys
from src.utilities.load_data import load_config

sys.path.append(os.getcwd())


if __name__ == '__main__':
    config = load_config('config_deploy.yaml')
    for app in config['running_apps']:
        if app['web_framework'] == 'flask':
            server_root = config['flask_server_root']
        elif app['web_framework'] == 'django':
            server_root = config['django_server_root']
        app_starter = config['deploy_root'] + os.path.sep + app['app_name'] + os.path.sep + server_root + os.path.sep + config['server_start_file']
        cmd = 'python ' + app_starter + '&&' + 'pause'
        os.system(cmd)