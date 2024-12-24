# -*- coding: utf-8 -*-
from utils import Registry

__all__ = ["AGENTS", "EMBRYOS", "MODELS", "build_agent", "build_embryo", "build_model"]


AGENTS = Registry("agent")
EMBRYOS = Registry("embryo")
MODELS = Registry("model")


def build_agent(cfg, device):
    if hasattr(cfg, "embryo"):
        for name in cfg.embryo.keys():
            if isinstance(cfg.embryo[name], dict) and hasattr(cfg.embryo[name], "type"):
                cfg.embryo[name]["device"] = device
                cfg.embryo[name] = build_model(cfg.embryo[name])
        cfg.embryo = build_embryo(cfg.embryo)
    else:
        raise AttributeError(f'agent must be has "embryo", but got {cfg}')
    return AGENTS.build(cfg)


def build_embryo(cfg):
    return EMBRYOS.build(cfg)


def build_model(cfg):
    m = MODELS.build(cfg)
    m.init_params()
    m.to_device()
    return m
