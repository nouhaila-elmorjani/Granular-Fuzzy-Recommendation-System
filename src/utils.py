import time
import logging
from functools import wraps
import pandas as pd

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('project.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def save_dataframe(df, filename, directory):
    """Save DataFrame to CSV with proper path handling"""
    from src.config import Config
    filepath = directory / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"Data saved to: {filepath}")
    return filepath

def load_dataframe(filename, directory):
    """Load DataFrame from CSV"""
    from src.config import Config
    filepath = directory / f"{filename}.csv"
    return pd.read_csv(filepath)