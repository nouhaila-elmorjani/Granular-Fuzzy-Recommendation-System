import pandas as pd
import numpy as np
from config import Config
from utils import setup_logging, timer

logger = setup_logging()

class GenreFuzzifier:
    """Convert binary genre classification to fuzzy membership values"""
    
    def __init__(self):
        self.config = Config()
        self.genre_relationships = self._build_genre_relationships()
    
    def _build_genre_relationships(self):
        """Define relationships between genres for enhanced blending"""
        relationships = {
            'Action': {'Adventure': 0.7, 'Thriller': 0.6, 'Sci-Fi': 0.5},
            'Adventure': {'Action': 0.7, 'Fantasy': 0.6, 'Romance': 0.4},
            'Comedy': {'Romance': 0.8, 'Drama': 0.5, 'Musical': 0.6},
            'Drama': {'Romance': 0.7, 'Comedy': 0.5, 'Thriller': 0.4},
            'Romance': {'Drama': 0.7, 'Comedy': 0.8, 'Musical': 0.5},
            'Thriller': {'Action': 0.6, 'Mystery': 0.7, 'Crime': 0.8},
            'Sci-Fi': {'Action': 0.5, 'Adventure': 0.6, 'Fantasy': 0.7},
            'Fantasy': {'Adventure': 0.6, 'Sci-Fi': 0.7, 'Animation': 0.5},
            'Horror': {'Thriller': 0.6, 'Mystery': 0.5, 'Fantasy': 0.3},
            'Mystery': {'Thriller': 0.7, 'Crime': 0.8, 'Drama': 0.4},
            'Crime': {'Thriller': 0.8, 'Drama': 0.6, 'Mystery': 0.8},
            'Animation': {"Children's": 0.9, 'Fantasy': 0.6, 'Comedy': 0.7},
            "Children's": {'Animation': 0.9, 'Fantasy': 0.5, 'Comedy': 0.6},
            'Documentary': {'Drama': 0.3},  # Documentaries are more distinct
            'Film-Noir': {'Crime': 0.7, 'Drama': 0.6, 'Mystery': 0.6},
            'Musical': {'Comedy': 0.6, 'Romance': 0.5, 'Drama': 0.4},
            'War': {'Drama': 0.8, 'Action': 0.5, 'Adventure': 0.4},
            'Western': {'Adventure': 0.7, 'Action': 0.5, 'Drama': 0.6}
        }
        return relationships
    
    @timer
    def binary_to_fuzzy(self, genre_vector):
        """Convert binary genre vector to fuzzy membership values"""
        fuzzy_vector = {}
        
        for genre in self.config.GENRES:
            if genre == 'unknown':
                continue
                
            if genre_vector[genre]:  # If genre is present
                # Primary genre gets high membership
                fuzzy_vector[genre] = np.random.uniform(0.7, 1.0)
                
                # Add related genres with lower membership
                if genre in self.genre_relationships:
                    for related_genre, strength in self.genre_relationships[genre].items():
                        if related_genre not in fuzzy_vector:  # Don't override primary
                            fuzzy_vector[related_genre] = np.random.uniform(0.2, 0.6) * strength
            else:
                # Genre not present, but might have small membership due to relationships
                fuzzy_vector[genre] = 0.0
        
        # Normalize to ensure values are between 0 and 1
        for genre in fuzzy_vector:
            fuzzy_vector[genre] = min(1.0, max(0.0, fuzzy_vector[genre]))
            
        return fuzzy_vector
    
    @timer
    def fuzzify_movie_dataframe(self, movies_df):
        """Convert entire movies dataframe to fuzzy genre representation"""
        fuzzy_movies = []
        
        for idx, movie in movies_df.iterrows():
            fuzzy_genres = self.binary_to_fuzzy(movie)
            
            fuzzy_movie = {
                'movie_id': movie['movie_id'],
                'title': movie['title'],
                **fuzzy_genres
            }
            fuzzy_movies.append(fuzzy_movie)
        
        # Create new dataframe with fuzzy genres
        fuzzy_df = pd.DataFrame(fuzzy_movies)
        
        # Fill missing genres with 0
        for genre in self.config.GENRES:
            if genre != 'unknown' and genre not in fuzzy_df.columns:
                fuzzy_df[genre] = 0.0
        
        logger.info(f"Fuzzified {len(fuzzy_df)} movies")
        return fuzzy_df
    
    def explain_fuzzification(self, movie_title, binary_genres, fuzzy_genres):
        """Generate human-readable explanation of fuzzification"""
        primary_genres = [genre for genre, value in binary_genres.items() 
                         if value and genre != 'unknown']
        
        enhanced_genres = {genre: value for genre, value in fuzzy_genres.items() 
                          if value > 0.3 and genre not in primary_genres}
        
        explanation = f"Movie: {movie_title}\n"
        explanation += f"Primary genres: {', '.join(primary_genres)}\n"
        
        if enhanced_genres:
            explanation += "Enhanced with: "
            enhanced_list = [f"{genre} ({value:.2f})" for genre, value in enhanced_genres.items()]
            explanation += ', '.join(enhanced_list)
        
        return explanation

# Test the fuzzifier
if __name__ == "__main__":
    print("Testing Genre Fuzzifier...")
    
    # Load data first
    from data_loader import MovieLensLoader
    loader = MovieLensLoader()
    data = loader.load_all_data()
    
    if data:
        fuzzifier = GenreFuzzifier()
        
        # Test on first 5 movies
        sample_movies = data['movies'].head()
        
        print("\n=== FUZZIFICATION EXAMPLES ===")
        for idx, movie in sample_movies.iterrows():
            fuzzy_genres = fuzzifier.binary_to_fuzzy(movie)
            
            # Get binary genres that are True
            binary_active = {genre: True for genre in Config.GENRES 
                           if movie[genre] and genre != 'unknown'}
            
            explanation = fuzzifier.explain_fuzzification(
                movie['title'], binary_active, fuzzy_genres
            )
            print(explanation)
            print("-" * 50)
        
        # Fuzzify entire dataset
        print("\nFuzzifying entire movie dataset...")
        fuzzy_movies = fuzzifier.fuzzify_movie_dataframe(data['movies'])
        print(f"Created fuzzy dataset with {len(fuzzy_movies)} movies")
        print("\nFirst movie fuzzy genres:")
        first_fuzzy = {k: v for k, v in fuzzy_movies.iloc[0].items() 
                      if k in Config.GENRES and v > 0}
        for genre, value in first_fuzzy.items():
            if value > 0.1:  # Only show significant memberships
                print(f"  {genre}: {value:.3f}")
    
    else:
        print("Failed to load data for fuzzification test")