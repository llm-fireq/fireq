import logging
from logging.handlers import RotatingFileHandler

def get_logger(name="mcomp", file_name="mcomp.log"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s::%(levelname)s::%(name)s::%(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = RotatingFileHandler(file_name, maxBytes=1024*1024, backupCount=1)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

    
def get_test_logger(name="test", file_name="test.log"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s::%(levelname)s::%(name)s::%(message)s')
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = RotatingFileHandler(file_name, maxBytes=1024*1024, backupCount=1)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger