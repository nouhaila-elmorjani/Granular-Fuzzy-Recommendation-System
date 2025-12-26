import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from config import Config

class GranularVisualization:
    """Visualize the granular computing process"""
    
    def __init__(self):
        self.config = Config()
    
    def plot_user_preference_evolution(self, drift_analysis):
        """Plot how user preferences change over time"""
        if not drift_analysis:
            print("No drift analysis data available")
            return None
            
        genres = list(drift_analysis.keys())
        early_strengths = [drift_analysis[g]['early'] for g in genres]
        recent_strengths = [drift_analysis[g]['recent'] for g in genres]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(genres))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], early_strengths, width, label='Early Preferences', alpha=0.7)
        ax.bar([i + width/2 for i in x], recent_strengths, width, label='Recent Preferences', alpha=0.7)
        
        ax.set_xlabel('Genres')
        ax.set_ylabel('Preference Strength')
        ax.set_title('User Preference Evolution Over Time')
        ax.set_xticks(x)
        ax.set_xticklabels(genres, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_user_genre_preferences(self, user_profiles, top_n=5):
        """Plot radar chart of user genre preferences"""
        if not user_profiles:
            print("No user profiles available")
            return None
            
        # Get top genres across all users
        all_genres = {}
        for profile in user_profiles:
            for genre, strength in profile['profile'].items():
                if strength > 0.1:  # Only significant preferences
                    all_genres[genre] = all_genres.get(genre, 0) + strength
        
        top_genres = sorted(all_genres.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_genre_names = [genre for genre, strength in top_genres]
        
        # Create radar chart data
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(top_genre_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, user_profile in enumerate(user_profiles[:3]):  # Plot first 3 users
            values = [user_profile['profile'].get(genre, 0) for genre in top_genre_names]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'User {user_profile["user_id"]}')
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_genre_names)
        ax.set_ylim(0, 1)
        ax.set_title('User Genre Preference Radar Chart')
        ax.legend(loc='upper right')
        
        return fig
    
    def plot_recommendation_similarity_heatmap(self, recommendations, user_profile):
        """Create heatmap of recommendation similarities"""
        if not recommendations:
            print("No recommendations available")
            return None
            
        # Extract data for heatmap
        movie_titles = [rec['title'][:20] + '...' if len(rec['title']) > 20 else rec['title'] 
                       for rec in recommendations[:10]]  # Top 10 recommendations
        
        genre_strengths = []
        genres_to_plot = sorted(user_profile['profile'].items(), 
                              key=lambda x: x[1], reverse=True)[:8]  # Top 8 user genres
        
        for rec in recommendations[:10]:
            movie_genres = {}
            if 'genres' in rec:
                movie_genres = dict(rec['genres'])
            
            row = [movie_genres.get(genre[0], 0) for genre in genres_to_plot]
            genre_strengths.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(genre_strengths, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(genres_to_plot)))
        ax.set_xticklabels([genre[0] for genre in genres_to_plot], rotation=45)
        ax.set_yticks(range(len(movie_titles)))
        ax.set_yticklabels(movie_titles)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Genre Strength', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(movie_titles)):
            for j in range(len(genres_to_plot)):
                text = ax.text(j, i, f'{genre_strengths[i][j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Recommendation Genre Alignment Heatmap')
        plt.tight_layout()
        
        return fig
    
    def plot_fuzzy_genre_distribution(self, fuzzy_movies_df, genre_sample=5):
        """Plot distribution of fuzzy genre memberships"""
        if fuzzy_movies_df.empty:
            print("No fuzzy movies data available")
            return None
            
        # Select sample of genres to plot
        available_genres = [g for g in self.config.GENRES if g != 'unknown' and g in fuzzy_movies_df.columns]
        genres_to_plot = available_genres[:genre_sample]
        
        fig, axes = plt.subplots(1, len(genres_to_plot), figsize=(15, 4))
        if len(genres_to_plot) == 1:
            axes = [axes]
        
        for i, genre in enumerate(genres_to_plot):
            genre_values = fuzzy_movies_df[genre]
            
            axes[i].hist(genre_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_xlabel('Membership Strength')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{genre} Distribution')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Simple test function
def test_visualization():
    """Test the visualization functions with sample data"""
    print("Testing visualization functions...")
    
    # Create sample data for testing
    sample_drift = {
        'Action': {'early': 0.3, 'recent': 0.6, 'drift': 0.3, 'trend': 'increasing'},
        'Comedy': {'early': 0.8, 'recent': 0.5, 'drift': -0.3, 'trend': 'decreasing'},
        'Drama': {'early': 0.4, 'recent': 0.4, 'drift': 0.0, 'trend': 'stable'}
    }
    
    sample_user_profiles = [
        {
            'user_id': 1,
            'profile': {'Action': 0.6, 'Comedy': 0.3, 'Drama': 0.8, 'Romance': 0.2}
        },
        {
            'user_id': 2, 
            'profile': {'Action': 0.2, 'Comedy': 0.9, 'Drama': 0.4, 'Thriller': 0.7}
        }
    ]
    
    sample_recommendations = [
        {
            'title': 'Sample Movie 1',
            'similarity_score': 0.75,
            'genres': [('Action', 0.8), ('Drama', 0.6), ('Thriller', 0.3)]
        },
        {
            'title': 'Sample Movie 2', 
            'similarity_score': 0.68,
            'genres': [('Comedy', 0.9), ('Romance', 0.7)]
        }
    ]
    
    # Test visualizations
    viz = GranularVisualization()
    
    try:
        # Test preference evolution plot
        fig1 = viz.plot_user_preference_evolution(sample_drift)
        if fig1:
            plt.show()
        
        # Test radar chart
        fig2 = viz.plot_user_genre_preferences(sample_user_profiles)
        if fig2:
            plt.show()
            
        print("Visualization test completed successfully!")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")

if __name__ == "__main__":
    test_visualization()