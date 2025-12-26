import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils import setup_logging, timer

logger = setup_logging()

class MovieLensLoader:
    """MovieLens 100K dataset loader and processor"""
    
    def __init__(self):
        self.config = Config()
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        
    def download_dataset(self):
        """Download MovieLens 100K dataset if not exists"""
        zip_path = self.config.RAW_DATA_DIR / "ml-100k.zip"
        
        if zip_path.exists():
            logger.info("Dataset already downloaded")
            return True
            
        try:
            logger.info("Downloading MovieLens 100K dataset...")
            response = requests.get(self.config.ML_100K_URL, stream=True)
            response.raise_for_status()
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            downloaded = 0
            
            with open(zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"Download progress: {percent:.1f}%", end='\r')
            
            print("\nExtracting dataset...")
            # Extract files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.RAW_DATA_DIR)
            
            logger.info("Dataset downloaded and extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    @timer
    def load_movies(self):
        """Load movies data with genre information"""
        try:
            # MovieLens 100K movies file has specific format
            movies_path = self.config.RAW_DATA_DIR / "ml-100k" / "u.item"
            
            # Define column names
            column_names = [
                'movie_id', 'title', 'release_date', 'video_release_date', 
                'imdb_url'
            ] + self.config.GENRES
            
            # Load data with specific separator and encoding
            self.movies_df = pd.read_csv(
                movies_path, 
                sep='|', 
                encoding='latin-1',
                header=None,
                names=column_names
            )
            
            # Convert genre columns to boolean
            for genre in self.config.GENRES:
                self.movies_df[genre] = self.movies_df[genre].astype(bool)
            
            logger.info(f"Loaded {len(self.movies_df)} movies")
            return self.movies_df
            
        except Exception as e:
            logger.error(f"Error loading movies: {e}")
            return None
    
    @timer
    def load_ratings(self):
        """Load ratings data"""
        try:
            ratings_path = self.config.RAW_DATA_DIR / "ml-100k" / "u.data"
            
            self.ratings_df = pd.read_csv(
                ratings_path,
                sep='\t',
                header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
            return self.ratings_df
            
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            return None
    
    @timer
    def load_users(self):
        """Load users data"""
        try:
            users_path = self.config.RAW_DATA_DIR / "ml-100k" / "u.user"
            
            self.users_df = pd.read_csv(
                users_path,
                sep='|',
                header=None,
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
            )
            
            logger.info(f"Loaded {len(self.users_df)} users")
            return self.users_df
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return None
    
    def get_data_summary(self):
        """Generate comprehensive data summary"""
        if self.movies_df is None or self.ratings_df is None:
            logger.warning("Please load data first")
            return None
            
        summary = {
            'total_movies': len(self.movies_df),
            'total_users': self.ratings_df['user_id'].nunique(),
            'total_ratings': len(self.ratings_df),
            'avg_ratings_per_user': len(self.ratings_df) / self.ratings_df['user_id'].nunique(),
            'avg_ratings_per_movie': len(self.ratings_df) / len(self.movies_df),
            'rating_distribution': self.ratings_df['rating'].value_counts().sort_index().to_dict(),
            'genre_distribution': {genre: self.movies_df[genre].sum() for genre in self.config.GENRES if genre != 'unknown'}
        }
        
        return summary
    
    def validate_data_quality(self):
        """Validate data quality and integrity"""
        issues = []
        
        # Check for missing values
        if self.movies_df.isnull().sum().sum() > 0:
            issues.append("Missing values in movies data")
        
        if self.ratings_df.isnull().sum().sum() > 0:
            issues.append("Missing values in ratings data")
            
        # Check rating range
        if not self.ratings_df['rating'].between(1, 5).all():
            issues.append("Ratings outside valid range (1-5)")
            
        # Check for duplicate ratings
        duplicate_ratings = self.ratings_df.duplicated(subset=['user_id', 'movie_id']).sum()
        if duplicate_ratings > 0:
            issues.append(f"Found {duplicate_ratings} duplicate user-movie ratings")
        
        return issues
    
    def load_all_data(self):
        """Load all datasets and return summary"""
        print("Starting data loading process...")
        success = self.download_dataset()
        if not success:
            print("Failed to download dataset")
            return None
            
        movies = self.load_movies()
        ratings = self.load_ratings()
        users = self.load_users()
        
        if movies is None or ratings is None:
            print("Failed to load essential data")
            return None
        
        summary = self.get_data_summary()
        issues = self.validate_data_quality()
        
        return {
            'movies': movies,
            'ratings': ratings,
            'users': users,
            'summary': summary,
            'quality_issues': issues
        }

# Test the data loader
if __name__ == "__main__":
    print("Testing MovieLens Data Loader...")
    loader = MovieLensLoader()
    data = loader.load_all_data()
    
    if data:
        print("\n=== DATA SUMMARY ===")
        for key, value in data['summary'].items():
            print(f"{key}: {value}")
        
        print("\n=== DATA QUALITY ISSUES ===")
        print(data['quality_issues'])
        
        print("\nFirst 5 movies:")
        print(data['movies'][['movie_id', 'title']].head())
        
        print("\nFirst 5 ratings:")
        print(data['ratings'].head())
    else:
        print("Failed to load data")