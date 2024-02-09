import logging


def set_logger(log_path):
    """
    Set up a logger for use.
    """
    # Config the logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Set level and formats
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Record logging
    logger.info(log_path)

    return logger