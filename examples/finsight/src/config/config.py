import json
import os
import re
import yaml


class Config:
    def __init__(self, config_file_path=None, config_dict={}):
        # load default config
        current_path = os.path.dirname(os.path.realpath(__file__))
        default_file_path = os.path.join(current_path, "default_config.yaml")
        self.config = self._load_config(default_file_path)

        # load from file
        self.config_file_path = config_file_path
        if config_file_path is not None:
            file_config = self._load_config(config_file_path)
            self.config.update(file_config)
        
        # load from dict
        self.config.update(config_dict)
        language_mapping = {
            'zh': 'Chinese (中文)',
            'en': 'English'
        }
        self.config.update({
            "language": language_mapping[self.config.get('language', 'zh')],
        })
        
        self._set_dirs()
    
    def _load_config(self, config_file_path):
        def build_yaml_loader():
            loader = yaml.FullLoader
            loader.add_implicit_resolver(
                "tag:yaml.org,2002:float",
                re.compile(
                    """^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
                    re.X,
                ),
                list("-+0123456789."),
            )
            return loader
    
        def replace_env_vars(obj):
            """Recursively replace ${VAR_NAME} with environment variables"""
            if isinstance(obj, dict):
                return {key: replace_env_vars(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Match ${VAR_NAME} pattern
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, obj)
                if matches:
                    result = obj
                    for var_name in matches:
                        env_value = os.getenv(var_name)
                        if env_value is None:
                            raise ValueError(f"Environment variable '{var_name}' is not set")
                        result = result.replace(f"${{{var_name}}}", env_value)
                    return result
                return obj
            else:
                return obj
    
        yaml_loader = build_yaml_loader()
        file_config = dict()
        if os.path.exists(config_file_path):
            if config_file_path.endswith('.yaml'):
                with open(config_file_path, "r", encoding="utf-8") as f:
                    file_config.update(yaml.load(f.read(), Loader=yaml_loader))
            elif config_file_path.endswith('.json'):
                with open(config_file_path, 'r') as f:
                    file_config.update(json.load(f))
            else:
                raise ValueError(f"Unsupported file type: {config_file_path}")
        else:
            raise ValueError(f"Config file not found: {config_file_path}")
        
        # Replace environment variables in the loaded config
        file_config = replace_env_vars(file_config)
        return file_config
    
    def _set_dirs(self):
        # convert output dir to absolute path
        output_dir = self.config.get('output_dir', './outputs')
        self.config['output_dir'] = output_dir
        target = self.config.get('target_name', 'unknown')
        save_note = self.config.get('save_note', None)
        target = target[:50]
        if save_note:
            target = str(save_note) + '_' + target
        self.working_dir = os.path.join(output_dir, target)
        self.config['working_dir'] = self.working_dir
        os.makedirs(self.working_dir, exist_ok=True)
        with open(os.path.join(self.working_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def __str__(self):
        return str(self.config)
