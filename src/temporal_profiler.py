class TemporalFuzzyProfiler:
    """Track how user preferences change over time"""
    
    def __init__(self):
        self.config = Config()
    
    def detect_preference_evolution(self, user_ratings):
        """Analyze how user preferences change over rating history"""
        # Sort ratings by timestamp
        sorted_ratings = user_ratings.sort_values('timestamp')
        
        # Split into time windows
        early_ratings = sorted_ratings.head(len(sorted_ratings)//3)
        recent_ratings = sorted_ratings.tail(len(sorted_ratings)//3)
        
        # Compare preference shifts
        early_profile = self._create_time_window_profile(early_ratings)
        recent_profile = self._create_time_window_profile(recent_ratings)
        
        # Calculate preference drift
        drift_analysis = {}
        for genre in self.config.GENRES:
            if genre != 'unknown':
                early_strength = early_profile.get(genre, 0)
                recent_strength = recent_profile.get(genre, 0)
                drift = recent_strength - early_strength
                drift_analysis[genre] = {
                    'early': early_strength,
                    'recent': recent_strength, 
                    'drift': drift,
                    'trend': 'increasing' if drift > 0.1 else 'decreasing' if drift < -0.1 else 'stable'
                }
        
        return drift_analysis
    
    def generate_temporal_recommendations(self, user_profile, drift_analysis, fuzzy_movies_df, top_n=10):
        """Recommend movies that align with user's evolving tastes"""
        recommendations = []
        
        for _, movie in fuzzy_movies_df.iterrows():
            # Base similarity
            base_score = self.calculate_similarity(user_profile, movie)
            
            # Temporal boost for genres user is trending toward
            temporal_boost = 0
            for genre, strength in self._get_movie_genres(movie).items():
                if genre in drift_analysis and drift_analysis[genre]['trend'] == 'increasing':
                    temporal_boost += drift_analysis[genre]['drift'] * strength
            
            final_score = base_score + (temporal_boost * 0.3)  # Weight temporal factor
            
            recommendations.append({
                'movie_id': movie['movie_id'],
                'title': movie['title'],
                'base_score': base_score,
                'temporal_boost': temporal_boost,
                'final_score': final_score,
                'reasoning': self._explain_temporal_recommendation(drift_analysis, movie)
            })
        
        return sorted(recommendations, key=lambda x: x['final_score'], reverse=True)[:top_n]