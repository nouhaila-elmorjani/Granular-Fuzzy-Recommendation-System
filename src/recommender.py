import pandas as pd
import numpy as np
from config import Config
from utils import setup_logging, timer

logger = setup_logging()

class FuzzySimilarity:
    """Calculate fuzzy similarity measures between user profiles and movies"""
    
    def __init__(self):
        self.config = Config()
    
    def fuzzy_jaccard(self, user_profile, movie_profile):
        """Fuzzy Jaccard similarity"""
        intersection = 0.0
        union = 0.0
        
        for genre in self.config.GENRES:
            if genre == 'unknown':
                continue
                
            user_val = user_profile.get(genre, 0.0)
            movie_val = movie_profile.get(genre, 0.0)
            
            intersection += min(user_val, movie_val)
            union += max(user_val, movie_val)
        
        return intersection / union if union > 0 else 0.0
    
    def fuzzy_cosine(self, user_profile, movie_profile):
        """Fuzzy Cosine similarity"""
        dot_product = 0.0
        user_norm = 0.0
        movie_norm = 0.0
        
        for genre in self.config.GENRES:
            if genre == 'unknown':
                continue
                
            user_val = user_profile.get(genre, 0.0)
            movie_val = movie_profile.get(genre, 0.0)
            
            dot_product += user_val * movie_val
            user_norm += user_val ** 2
            movie_norm += movie_val ** 2
        
        user_norm = np.sqrt(user_norm)
        movie_norm = np.sqrt(movie_norm)
        
        return dot_product / (user_norm * movie_norm) if user_norm > 0 and movie_norm > 0 else 0.0
    
    def fuzzy_dice(self, user_profile, movie_profile):
        """Fuzzy Dice coefficient"""
        intersection = 0.0
        sum_profiles = 0.0
        
        for genre in self.config.GENRES:
            if genre == 'unknown':
                continue
                
            user_val = user_profile.get(genre, 0.0)
            movie_val = movie_profile.get(genre, 0.0)
            
            intersection += min(user_val, movie_val)
            sum_profiles += user_val + movie_val
        
        return (2 * intersection) / sum_profiles if sum_profiles > 0 else 0.0
    
    def hybrid_similarity(self, user_profile, movie_profile, weights=None):
        """Hybrid similarity combining multiple measures"""
        if weights is None:
            weights = {'jaccard': 0.4, 'cosine': 0.4, 'dice': 0.2}
        
        jaccard = self.fuzzy_jaccard(user_profile, movie_profile)
        cosine = self.fuzzy_cosine(user_profile, movie_profile)
        dice = self.fuzzy_dice(user_profile, movie_profile)
        
        hybrid = (weights['jaccard'] * jaccard + 
                 weights['cosine'] * cosine + 
                 weights['dice'] * dice)
        
        return hybrid

class FuzzyRecommender:
    """Fuzzy logic-based movie recommendation system"""
    
    def __init__(self):
        self.config = Config()
        self.similarity = FuzzySimilarity()
    
    @timer
    def generate_recommendations(self, user_profile, fuzzy_movies_df, user_rated_movies, top_n=10):
        """Generate movie recommendations for a user"""
        recommendations = []
        
        user_id = user_profile['user_id']
        user_preferences = user_profile['profile']
        
        # Get movies the user hasn't rated
        unrated_movies = fuzzy_movies_df[~fuzzy_movies_df['movie_id'].isin(user_rated_movies)]
        
        logger.info(f"Generating recommendations for user {user_id} from {len(unrated_movies)} unrated movies")
        
        for _, movie in unrated_movies.iterrows():
            # Create movie profile from fuzzy genres
            movie_profile = {}
            for genre in self.config.GENRES:
                if genre != 'unknown' and genre in movie:
                    movie_profile[genre] = movie[genre]
            
            # Calculate similarity
            similarity_score = self.similarity.hybrid_similarity(user_preferences, movie_profile)
            
            recommendations.append({
                'movie_id': movie['movie_id'],
                'title': movie['title'],
                'similarity_score': similarity_score,
                'genres': self._get_top_movie_genres(movie_profile)
            })
        
        # Sort by similarity score and return top N
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return recommendations[:top_n]
    
    def _get_top_movie_genres(self, movie_profile, n=3):
        """Get top genres for a movie with their strengths"""
        sorted_genres = sorted(movie_profile.items(), key=lambda x: x[1], reverse=True)
        return [(genre, strength) for genre, strength in sorted_genres[:n] if strength > 0.1]
    
    def _generate_explanation(self, user_profile, movie, similarity_score, top_genres):
        """Generate explanation for why a movie was recommended"""
        user_top_genres = sorted(user_profile['profile'].items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"Recommended '{movie}' (Score: {similarity_score:.3f})\n"
        explanation += f"Your top preferences: {', '.join([f'{g}({s:.2f})' for g, s in user_top_genres])}\n"
        explanation += f"Movie genres: {', '.join([f'{g}({s:.2f})' for g, s in top_genres])}"
        
        return explanation
    
    @timer
    def generate_diverse_recommendations(self, user_profile, fuzzy_movies_df, user_rated_movies, 
                                      top_n=10, diversity_factor=0.3):
        """Generate diverse recommendations using MMR algorithm"""
        recommendations = self.generate_recommendations(
            user_profile, fuzzy_movies_df, user_rated_movies, top_n=50
        )
        
        if not recommendations:
            return []
        
        # Maximal Marginal Relevance (MMR) for diversity
        diverse_recommendations = []
        remaining_recommendations = recommendations.copy()
        
        # Add the most relevant item first
        diverse_recommendations.append(remaining_recommendations.pop(0))
        
        while len(diverse_recommendations) < top_n and remaining_recommendations:
            best_score = -1
            best_index = -1
            
            for i, candidate in enumerate(remaining_recommendations):
                # Calculate relevance score
                relevance = candidate['similarity_score']
                
                # Calculate diversity score (max similarity to already selected items)
                max_similarity = 0
                for selected in diverse_recommendations:
                    # Simple genre overlap as diversity measure
                    selected_genres = set([g for g, s in selected['genres']])
                    candidate_genres = set([g for g, s in candidate['genres']])
                    overlap = len(selected_genres.intersection(candidate_genres))
                    similarity = overlap / max(len(selected_genres), len(candidate_genres))
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = relevance - diversity_factor * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = i
            
            if best_index >= 0:
                diverse_recommendations.append(remaining_recommendations.pop(best_index))
            else:
                break
        
        return diverse_recommendations[:top_n]

# Test the recommendation system
if __name__ == "__main__":
    print("Testing Fuzzy Recommendation System...")
    
    # Load data and create profiles
    from data_loader import MovieLensLoader
    from fuzzifier import GenreFuzzifier
    from user_profiler import FuzzyUserProfiler
    
    loader = MovieLensLoader()
    data = loader.load_all_data()
    
    if data:
        # Fuzzify movies
        fuzzifier = GenreFuzzifier()
        fuzzy_movies = fuzzifier.fuzzify_movie_dataframe(data['movies'])
        
        # Create user profiles (small sample for testing)
        profiler = FuzzyUserProfiler()
        user_profiles = profiler.create_all_profiles(
            data['ratings'], fuzzy_movies, sample_size=5
        )
        
        # Initialize recommender
        recommender = FuzzyRecommender()
        
        print(f"\n=== RECOMMENDATION EXAMPLES ===")
        
        for user_profile in user_profiles[:2]:  # Test with first 2 users
            user_id = user_profile['user_id']
            
            # Get movies this user has already rated
            user_rated_movies = data['ratings'][data['ratings']['user_id'] == user_id]['movie_id'].tolist()
            
            print(f"\n🎬 Recommendations for User {user_id}:")
            print(f"User preferences: {profiler.get_top_genres(user_profile, n=3)}")
            
            # Generate regular recommendations
            recommendations = recommender.generate_recommendations(
                user_profile, fuzzy_movies, user_rated_movies, top_n=5
            )
            
            print(f"\nTop 5 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
                print(f"     Genres: {rec['genres']}")
            
            # Generate diverse recommendations
            diverse_recommendations = recommender.generate_diverse_recommendations(
                user_profile, fuzzy_movies, user_rated_movies, top_n=5
            )
            
            print(f"\nDiverse Recommendations:")
            for i, rec in enumerate(diverse_recommendations, 1):
                print(f"  {i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")
                print(f"     Genres: {rec['genres']}")
            
            print("-" * 60)
    
    else:
        print("Failed to load data for recommendation test")