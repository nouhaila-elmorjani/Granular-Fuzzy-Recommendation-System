class MultiModalGranularRecommender:
    """Fuse multiple granular perspectives: genres, ratings, temporal patterns"""
    
    def __init__(self):
        self.config = Config()
    
    def extract_rating_pattern_granules(self, user_ratings):
        """Granulate users based on rating behavior patterns"""
        rating_stats = {
            'rating_mean': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'rating_entropy': self._calculate_rating_entropy(user_ratings),
            'harshness_score': self._calculate_harshness(user_ratings),
            'diversity_preference': self._calculate_rating_diversity(user_ratings)
        }
        
        # Create fuzzy granules for rating behavior
        behavior_profile = {
            'harsh_critic': max(0, (2.5 - rating_stats['rating_mean']) / 2.5),
            'easy_grader': max(0, (rating_stats['rating_mean'] - 3.5) / 1.5),
            'consistent_rater': 1 - min(1, rating_stats['rating_std'] / 2),
            'diverse_tastes': rating_stats['diversity_preference']
        }
        
        return behavior_profile
    
    def hybrid_granular_similarity(self, user1_profile, user2_profile, user1_behavior, user2_behavior):
        """Combine genre similarity with behavioral similarity"""
        genre_similarity = self.calculate_genre_similarity(user1_profile, user2_profile)
        behavior_similarity = self.calculate_behavior_similarity(user1_behavior, user2_behavior)
        
        # Adaptive weighting based on user consistency
        user1_consistency = user1_behavior['consistent_rater']
        user2_consistency = user2_behavior['consistent_rater']
        behavior_weight = (user1_consistency + user2_consistency) / 2
        
        final_similarity = (genre_similarity * (1 - behavior_weight)) + (behavior_similarity * behavior_weight)
        
        return final_similarity
    
    def generate_context_aware_recommendations(self, user_id, fuzzy_movies_df, ratings_df, context='weekend'):
        """Context-aware recommendations based on time/season"""
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        if context == 'weekend':
            # Recommend more action/comedy for weekends
            context_boost = {'Action': 0.2, 'Comedy': 0.15, 'Adventure': 0.1}
        elif context == 'weeknight':
            # Recommend drama/documentary for weeknights
            context_boost = {'Drama': 0.2, 'Documentary': 0.15, 'Mystery': 0.1}
        else:
            context_boost = {}
        
        # Apply context boosting to recommendations
        recommendations = []
        for _, movie in fuzzy_movies_df.iterrows():
            base_score = self.calculate_similarity(user_profile, movie)
            
            # Apply context boost
            context_score = 0
            for genre, boost in context_boost.items():
                if genre in movie and movie[genre] > 0.3:
                    context_score += boost * movie[genre]
            
            final_score = base_score + context_score
            
            recommendations.append({
                'movie_id': movie['movie_id'],
                'title': movie['title'],
                'base_score': base_score,
                'context_boost': context_score,
                'final_score': final_score,
                'context': context
            })
        
        return sorted(recommendations, key=lambda x: x['final_score'], reverse=True)[:10]