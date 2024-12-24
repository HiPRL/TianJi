# -*- coding: utf-8 -*-
import logging



__all__ = ["root_logger", "make_logger", "print_log"]


class Logger(object):
    def __init__(self):
        self.logger_initialized = {}
    
    def make_logger(self, name, log_file=None, log_level=logging.INFO, file_mode='w', disable=False):
        logger = logging.getLogger(name)
        if name in self.logger_initialized:
            return logger

        for logger_name in self.logger_initialized:
            if name.startswith(logger_name):
                return logger

        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

        stream_handler = logging.StreamHandler()
        handlers = [stream_handler]

        if log_file is not None and not disable:
            import os.path as osp
            from pathlib import Path
            Path(osp.dirname(log_file)).mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, file_mode)
            handlers.append(file_handler)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

        logger.disabled = disable
        logger.setLevel(log_level)

        self.logger_initialized[name] = True

        return logger
    
    def print_log(self, msg, logger=None, level=logging.INFO):
        if logger is None:
            print(msg)
        elif isinstance(logger, logging.Logger):
            logger.log(level, msg)
        elif logger == 'silent':
            pass
        elif isinstance(logger, str):
            _logger = get_logger(logger)
            _logger.log(level, msg)
        else:
            raise TypeError(f'logger should be either in [logging.Logger object, str, "silent", None], but got {type(logger)}')
    
    def root_logger(self, log_file=None, log_level=logging.INFO):
        return self.make_logger(name='tianji', log_file=log_file, log_level=log_level)



logger = Logger()
root_logger = logger.root_logger()
make_logger = logger.make_logger
print_log = logger.print_log