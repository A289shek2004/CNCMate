# src/utils.py
import os
import logging

def setup_logging(log_dir="logs", log_file="app.log"):
    """
    Sets up basic logging configuration.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
