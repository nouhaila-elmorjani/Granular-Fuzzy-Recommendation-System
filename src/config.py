import os
from pathlib import Path

class Config:
    """Configuration management for the project"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # MovieLens dataset URLs
    ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    # File names
    RATINGS_FILE = "u.data"
    MOVIES_FILE = "u.item"
    USERS_FILE = "u.user"
    
    # Genre information (MovieLens 100K has 19 genres)
    GENRES = [
        'unknown', 'Action', 'Adventure', 'Animation', 
        'Children\'s', 'Comedy', 'Crime', 'Documentary', 
        'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
        'Thriller', 'War', 'Western'
    ]
    
    # Visualization settings
    VISUALIZATION_SETTINGS = {
        'figure_size': (12, 8),
        'color_palette': 'viridis',
        'style': 'seaborn'
    }
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Initialize directories when module is imported
Config.setup_directories()