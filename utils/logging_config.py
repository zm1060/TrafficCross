import logging
from pathlib import Path
from datetime import datetime

def setup_logging(model_name, dataset_name, log_dir='logs', seed=None):
    """Configure logging.

    Args:
        model_name (str): Model name.
        dataset_name (str): Dataset name.
        log_dir (str): Log directory.
        seed (int): Optional training seed included in the log filename.

    Returns:
        logger: Configured logger instance.
    """
    # Create the log directory.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamp.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build the log filename.
    seed_suffix = f'_seed_{seed}' if seed is not None else ''
    log_filename = f"{model_name}_{dataset_name}{seed_suffix}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Create the logger.
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # Disable propagation to the root logger to avoid duplicate output.
    logger.propagate = False

    # Remove existing handlers.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create the file handler.
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Create the console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create the formatter.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach handlers.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Started logging to {log_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    if seed is not None:
        logger.info(f"Training seed: {seed}")

    return logger
