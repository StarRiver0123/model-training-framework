import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_django_manage_file():
    server_root = os.path.dirname(os.path.abspath(__file__))
    server_paths = []
    server_paths.append(server_root)
    while len(server_paths):
        f_p = server_paths.pop()
        if os.path.isfile(f_p):
            if f_p.split(os.path.sep)[-1] == 'manage.py':
                django_project_name = f_p.split(os.path.sep)[-2]
                break
        if os.path.isdir(f_p):
            for sub_f_p in os.listdir(f_p):
                server_paths.append(f_p + os.path.sep + sub_f_p)

    return server_root + os.path.sep + django_project_name + os.path.sep + 'manage.py'


django_manage_file = find_django_manage_file()
cmd = 'python ' + django_manage_file + ' runserver 8080' + '&&' + 'pause'
os.system(cmd)


