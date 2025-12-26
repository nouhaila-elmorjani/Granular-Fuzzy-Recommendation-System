import pandas as pd
import numpy as np
from config import Config

class ExplainableFuzzyRecommender:
    """Generate human-readable explanations for recommendations"""
    
    def __init__(self):
        self.config = Config()
    
    def generate_detailed_explanation(self, user_profile, movie, similarity_score, top_genres):
        """Generate comprehensive explanation for recommendation"""
        user_top_genres = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation_parts = []
        
        # Genre alignment explanation
        genre_matches = []
        for user_genre, user_strength in user_top_genres:
            movie_genres_dict = dict(top_genres)
            if user_genre in movie_genres_dict:
                movie_strength = movie_genres_dict[user_genre]
                match_strength = min(user_strength, movie_strength)
                if match_strength > 0.3:
                    genre_matches.append((user_genre, match_strength))
        
        if genre_matches:
            match_text = ', '.join([f"{g[0]}" for g in genre_matches])
            explanation_parts.append(f"Matches your love for {match_text}")
        
        # Strength explanation
        if similarity_score > 0.7:
            strength_msg = "Strong match with your preferences"
        elif similarity_score > 0.5:
            strength_msg = "Good alignment with your tastes"
        else:
            strength_msg = "Exploratory recommendation based on related interests"
        explanation_parts.append(strength_msg)
        
        # Diversity explanation
        user_genre_names = set([g[0] for g in user_top_genres])
        movie_genre_names = set([g[0] for g in top_genres])
        unique_genres = movie_genre_names - user_genre_names
        
        if unique_genres:
            explanation_parts.append(f"Introduces new genres: {', '.join(unique_genres)}")
        
        return " | ".join(explanation_parts)
    
    def generate_comparative_explanation(self, user_profile, recommendations):
        """Explain the ranking of top recommendations"""
        if not recommendations:
            return "No recommendations available"
        
        user_top_genres = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        user_top_str = ', '.join([f"{g}({s:.2f})" for g, s in user_top_genres])
        
        explanation = f"Your Top Preferences: {user_top_str}\n\n"
        explanation += "Why these movies were recommended:\n\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            explanation += f"{i}. {rec['title']} (Score: {rec['similarity_score']:.3f})\n"
            
            # Show genre alignment
            movie_genres = dict(rec['genres'][:3])
            alignment = []
            for user_genre, user_strength in user_top_genres:
                if user_genre in movie_genres:
                    alignment.append(f"{user_genre}: {movie_genres[user_genre]:.2f}")
            
            if alignment:
                explanation += f"   - Genre alignment: {', '.join(alignment)}\n"
            
            # Explain ranking position
            if i == 1:
                explanation += "   - Top choice: Best overall genre match\n"
            elif i <= 3:
                explanation += "   - Strong alternative: Good match with some diversity\n"
            
            explanation += "\n"
        
        return explanation
    
    def generate_user_insights(self, user_profile, ratings_count, avg_rating):
        """Generate insights about user's watching behavior"""
        insights = []
        
        # Analyze preference strength
        top_genre, top_strength = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[0]
        
        if top_strength > 0.6:
            insights.append(f"You have a strong preference for {top_genre} content")
        elif top_strength > 0.3:
            insights.append(f"You enjoy {top_genre} movies among other genres")
        
        # Analyze rating behavior
        if avg_rating > 4.0:
            insights.append("You're generally positive in your ratings")
        elif avg_rating < 3.0:
            insights.append("You have high standards for movies")
        
        if ratings_count > 100:
            insights.append("You're an experienced movie watcher")
        elif ratings_count < 20:
            insights.append("We're still learning your preferences")
        
        return insights

# Test the explainable recommender
def test_explainable_features():
    """Test the explainable features"""
    print("Testing Explainable AI Features...")
    
    # Sample data
    sample_user_profile = {
        'Comedy': 0.67,
        'Drama': 0.29, 
        'Romance': 0.23,
        'Action': 0.15
    }
    
    sample_recommendations = [
        {
            'title': 'Boys on the Side (1995)',
            'similarity_score': 0.686,
            'genres': [('Comedy', 0.85), ('Drama', 0.72), ('Romance', 0.68)]
        },
        {
            'title': 'Waiting to Exhale (1995)',
            'similarity_score': 0.683,
            'genres': [('Drama', 0.78), ('Comedy', 0.65), ('Romance', 0.61)]
        }
    ]
    
    explainer = ExplainableFuzzyRecommender()
    
    # Test detailed explanation
    explanation = explainer.generate_detailed_explanation(
        sample_user_profile, 
        sample_recommendations[0],
        sample_recommendations[0]['similarity_score'],
        sample_recommendations[0]['genres']
    )
    print("Detailed Explanation:")
    print(f"   {explanation}")
    
    # Test comparative explanation
    comparative = explainer.generate_comparative_explanation(
        sample_user_profile, 
        sample_recommendations
    )
    print("\nComparative Explanation:")
    print(comparative)
    
    # Test user insights
    insights = explainer.generate_user_insights(sample_user_profile, 45, 3.8)
    print("\nUser Insights:")
    for insight in insights:
        print(f"   • {insight}")

if __name__ == "__main__":
    test_explainable_features()