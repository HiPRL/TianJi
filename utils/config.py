# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import os.path as osp
import sys
import re
import ast
import json
import types
import tempfile
from addict import Dict
from pathlib import Path
from io import BytesIO, StringIO
from importlib import import_module
from utils.common import check_file_exist

import yaml
try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader, Dumper



__all__ = ['Config']



class BaseFileHandler(metaclass=ABCMeta):

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath: str, mode: str = 'r', **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath: str, mode: str = 'w', **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)


class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('default', JsonHandler.set_default)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('default', JsonHandler.set_default)
        return json.dumps(obj, **kwargs)
    
    @staticmethod
    def set_default(obj):
        import numpy as np
        if isinstance(obj, (set, range)):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f'{type(obj)} is unsupported for json dump')


class YamlHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        return yaml.dump(obj, **kwargs)


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)
    
    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config:

    file_handlers = {
        'json': JsonHandler(),
        'yaml': YamlHandler(),
        'yml': YamlHandler()
    }

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, but got '{type(cfg_dict)}'")
        else:
            pass
        
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename) as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)
        super().__setattr__('_filename', filename)
        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
    
    def __repr__(self):
        return self._cfg_dict.__repr__()

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)
    
    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)
    
    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)
    
    def __iter__(self):
        return iter(self._cfg_dict)

    @staticmethod
    def fromfile(filepath):
        filename = str(filepath) if isinstance(filepath, Path) else filepath
        cfg_dict, cfg_text = Config._file2dict(filename)
        return Config(cfg_dict=cfg_dict, cfg_text=cfg_text, filename=filename)
    
    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if osp.splitext(filename)[1] not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only supported [py, json, yaml, yml] now!')
        
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=osp.splitext(filename)[1])
            temp_config_name = osp.basename(temp_config_file.name)
            Config._substitute_predefined_vars(filename, temp_config_file.name)
            if filename.endswith('.py'):
                Config._py_syntax_check(filename)
                sys.path.insert(0, temp_config_dir)
                mod = import_module(osp.splitext(temp_config_name)[0])
                sys.path.pop(0)
                cfg_dict = {
                    name:value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                    and not isinstance(value, types.ModuleType)
                    and not isinstance(value, types.FunctionType)
                }
                del sys.modules[osp.splitext(temp_config_name)[0]]
            elif filename.endswith(('.json', '.yaml', '.yml')):
                cfg_dict = Config._load(temp_config_file.name)
            temp_config_file.close()
        
        cfg_text = filename + '\n'
        with open(filename, encoding='utf-8') as f:
            cfg_text += f.read()
        return cfg_dict, cfg_text
    
    @staticmethod
    def _py_syntax_check(filename):
        with open(filename, encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError(f"config has syntax errors. file {filename}: {e}")
    
    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
    
    @staticmethod
    def _load(file, file_format=None, **kwargs):
        file = str(file) if isinstance(file, Path) else file
        if file_format is None and isinstance(file, str):
            file_format = file.split('.')[-1]
        if file_format not in Config.file_handlers:
            raise TypeError(f'Unsupported format: {file_format}')
        
        handler = Config.file_handlers[file_format]
        if isinstance(file, str):
            with open(file, encoding='utf-8') as fp:
                with StringIO(fp.read()) as f:
                    obj = handler.load_from_fileobj(f, **kwargs)
        elif hasattr(file, 'read'):
            obj = handler.load_from_fileobj(file, **kwargs)
        else:
            raise TypeError('"file" must be a filepath str or a file-object')
        return obj