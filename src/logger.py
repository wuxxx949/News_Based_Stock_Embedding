import os
import logging

from src.meta_data import get_meta_data

def setup_logger(
    logger_name: str,
    log_file: str,
    level=logging.INFO,
    rm_existing_log: bool = False
    ) -> logging.Logger:
    """setup logger
    """
    log_dir = get_meta_data()['LOG_DIR']
    if rm_existing_log:
        os.rmdir(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    stream_handler.setFormatter(formatter)

    log_file = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file, 'a+')
    formatter_with_date = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)-8s %(message)s',
        '%Y-%m-%d %H:%M:%S'
        )
    file_handler.setFormatter(formatter_with_date)
    file_handler.setLevel(level)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger