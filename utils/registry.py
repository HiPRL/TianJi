# -*- coding: utf-8 -*-
import inspect



__all__ = ['Registry']



class Registry:
    def __init__(self, name, build_func=None):
        self._name = name
        self._module_dict = dict()
        self.build_func = Registry.default_build if build_func is None else build_func
    
    def __repr__(self):
        return f"{self.__class__.__name__} '(name={self._name}, items={self._module_dict})'"
    
    def __len__(self):
        return len(self._module_dict)
    
    def __contains__(self, key):
        return self.get(key) is not None
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def get(self, key):
        if key in self._module_dict:
            return self._module_dict[key]
        else:
            return None
    
    def register_module(self, name=None, module=None):
        if module is not None:
            self._register_module(module=module, module_name=name)
            return module
        else:
            def _register(module):
                self._register_module(module=module, module_name=name)
                return module
            return _register

    def _register_module(self, module, module_name=None):
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError(f'module must be class or function, but got {type(module)}')
        
        module_name = module.__name__ if module_name is None else module_name
        self._module_dict[module_name] = module
    
    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    @staticmethod
    def default_build(cfg: dict, registry: 'Registry'):
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(f'cfg must contain the key "type", but got {cfg}')
        if not isinstance(registry, Registry):
            raise TypeError(f'registry must be an Registry object, but got {type(registry)}')

        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(f'{obj_type} is not in the {registry.name} registry')
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')