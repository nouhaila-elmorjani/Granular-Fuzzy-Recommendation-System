import pandas as pd
import numpy as np
from config import Config
from utils import setup_logging, timer

logger = setup_logging()

class FuzzyUserProfiler:
    """Create fuzzy user profiles based on movie ratings and fuzzy genres"""
    
    def __init__(self):
        self.config = Config()
    
    @timer
    def create_user_profile(self, user_id, ratings_df, fuzzy_movies_df):
        """Create fuzzy profile for a single user"""
        # Get user's ratings
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            logger.warning(f"No ratings found for user {user_id}")
            return None
        
        # Merge with fuzzy movies to get genre information
        user_data = user_ratings.merge(fuzzy_movies_df, on='movie_id')
        
        # Initialize user profile
        user_profile = {genre: 0.0 for genre in self.config.GENRES if genre != 'unknown'}
        
        # Calculate weighted average of fuzzy genres based on ratings
        total_weight = 0
        
        for _, row in user_data.iterrows():
            rating = row['rating']
            weight = (rating - 1) / 4.0  # Normalize rating to 0-1 scale
            
            for genre in user_profile.keys():
                if genre in row:
                    user_profile[genre] += row[genre] * weight
            
            total_weight += weight
        
        # Normalize the profile
        if total_weight > 0:
            for genre in user_profile.keys():
                user_profile[genre] /= total_weight
        
        return {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'average_rating': user_ratings['rating'].mean(),
            'profile': user_profile
        }
    
    @timer
    def create_all_profiles(self, ratings_df, fuzzy_movies_df, sample_size=None):
        """Create fuzzy profiles for all users"""
        user_ids = ratings_df['user_id'].unique()
        
        if sample_size:
            user_ids = user_ids[:sample_size]
            logger.info(f"Creating profiles for sample of {sample_size} users")
        else:
            logger.info(f"Creating profiles for all {len(user_ids)} users")
        
        user_profiles = []
        
        for user_id in user_ids:
            profile = self.create_user_profile(user_id, ratings_df, fuzzy_movies_df)
            if profile:
                user_profiles.append(profile)
        
        logger.info(f"Created {len(user_profiles)} user profiles")
        return user_profiles
    
    def get_top_genres(self, user_profile, n=5):
        """Get top N genres for a user profile"""
        profile_dict = user_profile['profile']
        sorted_genres = sorted(profile_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres[:n]
    
    def analyze_user_preferences(self, user_profiles):
        """Analyze overall user preference patterns"""
        genre_sums = {genre: 0.0 for genre in self.config.GENRES if genre != 'unknown'}
        genre_counts = {genre: 0 for genre in self.config.GENRES if genre != 'unknown'}
        
        for profile in user_profiles:
            for genre, value in profile['profile'].items():
                if value > 0.1:  # Only count significant preferences
                    genre_sums[genre] += value
                    genre_counts[genre] += 1
        
        # Calculate average preference strength per genre
        genre_analysis = {}
        for genre in genre_sums.keys():
            if genre_counts[genre] > 0:
                genre_analysis[genre] = {
                    'avg_strength': genre_sums[genre] / len(user_profiles),
                    'user_coverage': genre_counts[genre] / len(user_profiles),
                    'total_users': genre_counts[genre]
                }
        
        return genre_analysis

# Test the user profiler
if __name__ == "__main__":
    print("Testing Fuzzy User Profiler...")
    
    # Load data
    from data_loader import MovieLensLoader
    from fuzzifier import GenreFuzzifier
    
    loader = MovieLensLoader()
    data = loader.load_all_data()
    
    if data:
        # Fuzzify movies first
        fuzzifier = GenreFuzzifier()
        fuzzy_movies = fuzzifier.fuzzify_movie_dataframe(data['movies'])
        
        # Create user profiles
        profiler = FuzzyUserProfiler()
        
        # Test with first 10 users (for quick testing)
        user_profiles = profiler.create_all_profiles(
            data['ratings'], 
            fuzzy_movies, 
            sample_size=10
        )
        
        print(f"\n=== USER PROFILE EXAMPLES ===")
        for i, profile in enumerate(user_profiles[:3]):  # Show first 3 users
            print(f"\nUser {profile['user_id']}:")
            print(f"  Ratings: {profile['total_ratings']}")
            print(f"  Avg Rating: {profile['average_rating']:.2f}")
            
            top_genres = profiler.get_top_genres(profile, n=3)
            print("  Top Genres:")
            for genre, strength in top_genres:
                print(f"    {genre}: {strength:.3f}")
        
        # Analyze overall preferences
        print(f"\n=== OVERALL USER PREFERENCE ANALYSIS ===")
        analysis = profiler.analyze_user_preferences(user_profiles)
        
        print("Most popular genres across users:")
        sorted_analysis = sorted(analysis.items(), 
                               key=lambda x: x[1]['user_coverage'], 
                               reverse=True)
        
        for genre, stats in sorted_analysis[:5]:
            print(f"  {genre}: {stats['user_coverage']:.1%} users "
                  f"(avg strength: {stats['avg_strength']:.3f})")
    02
    else:
        print("Failed to load data for user profiling test")