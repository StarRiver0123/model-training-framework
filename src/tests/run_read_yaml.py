import yaml

with open('G:\AI\projects\mtf_projects\config_project.yaml', 'r', encoding='utf-8') as f:
    args = yaml.safe_load(f.read())

print(args)