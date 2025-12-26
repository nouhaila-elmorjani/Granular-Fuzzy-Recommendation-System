# Granular Fuzzy Recommendation System


A sophisticated movie recommendation system implementing **fuzzy logic** and **granular computing** principles to model nuanced user preferences and generate personalized, explainable recommendations.

---

## Abstract

This project implements a complete recommendation pipeline that transforms traditional binary genre classifications into **granular fuzzy representations**. By applying fuzzy set theory and granular computing techniques, the system captures the continuous nature of genre memberships and user preferences, enabling more accurate, diverse, and interpretable movie recommendations.

The system demonstrates the practical application of **soft computing techniques** to real-world recommendation problems using the **MovieLens 100K** dataset.

---

## Project Overview

The **Granular Fuzzy Recommendation System** addresses key limitations of traditional recommender systems by:

* Converting binary genre indicators into continuous fuzzy membership values
* Modeling semantic relationships between genres using fuzzy logic
* Constructing granular, rating-weighted user preference profiles
* Generating explainable recommendations with clear preference alignment
* Evaluating system performance using both technical and analytical metrics

---

## Key Innovations

* **Fuzzy Genre Representation** – Continuous membership values in the range [0, 1]
* **Genre Relationship Modeling** – Semantic similarity graph between genres
* **Rating-Weighted User Profiling** – Preferences scaled by user ratings
* **Hybrid Similarity Metrics** – Combination of Jaccard, Cosine, and Dice measures
* **MMR-Based Diversification** – Balancing relevance and recommendation diversity
* **Granular Computing Framework** – Multi-level abstraction of user preferences

---

## Technical Architecture

The system follows a modular pipeline:

```
Data Layer → Processing Layer → Recommendation Layer → Evaluation Layer
```

* **Data Layer**: Dataset ingestion and preprocessing
* **Processing Layer**: Genre fuzzification and user profile construction
* **Recommendation Layer**: Similarity computation and recommendation ranking
* **Evaluation Layer**: Performance analysis and visualization

---

## System Components

### Core Modules

* `data_loader.py` – MovieLens dataset loading and preprocessing
* `fuzzifier.py` – Binary-to-fuzzy genre conversion
* `user_profiler.py` – Granular fuzzy user preference modeling
* `recommender.py` – Fuzzy similarity computation and recommendation logic
* `evaluator.py` – Performance evaluation metrics
* `visualization.py` – Analytical visualizations

### Configuration & Utilities

* `config.py` – Centralized configuration
* `utils.py` – Logging, timing, and helper utilities

---

## Methodology

### 1. Genre Fuzzification

Binary genre labels are transformed into fuzzy membership values using:

* **Primary genre membership**: 0.7 – 1.0
* **Secondary genre propagation**: 0.2 – 0.6 scaled by semantic similarity
* **Normalization**: Ensures all memberships remain within [0, 1]

### 2. User Profile Construction

* Aggregation of fuzzy genre vectors from rated movies
* Rating-weighted contribution of each movie
* Normalization for cross-user comparability
* Extraction of dominant preference granules

### 3. Recommendation Generation

* Hybrid fuzzy similarity scoring
* Exclusion of previously rated items
* Maximal Marginal Relevance (MMR) for diversity control

### 4. Evaluation Strategy

* Similarity score distribution
* Personalization strength across users
* Genre alignment between recommendations and profiles
* Coverage and novelty analysis

---

## Dataset

The system uses the **MovieLens 100K** dataset:

* 100,000 ratings
* 943 users
* 1,682 movies
* 19 genres
* Data sparsity: ~93.7%

---

## Installation

### Requirements

* Python 3.8 or later
* pip package manager
* ≥ 4 GB RAM

## Usage

### Example

```python
from src.data_loader import MovieLensLoader
from src.fuzzifier import GenreFuzzifier
from src.user_profiler import FuzzyUserProfiler
from src.recommender import FuzzyRecommender

loader = MovieLensLoader()
fuzzifier = GenreFuzzifier()
profiler = FuzzyUserProfiler()
recommender = FuzzyRecommender()

data = loader.load_all_data()
fuzzy_movies = fuzzifier.fuzzify_movie_dataframe(data["movies"])
user_profiles = profiler.create_all_profiles(data["ratings"], fuzzy_movies)

recommendations = recommender.generate_recommendations(
    user_profiles[0], fuzzy_movies, top_n=10
)
```

---

## Jupyter Notebook Workflow

* `01_data_exploration.ipynb` – Dataset analysis
* `02_genre_fuzzification.ipynb` – Fuzzy representation
* `03_user_profiling_with_fuzzy_preferences.ipynb` – User modeling
* `04_similarity_and_recommendation.ipynb` – Recommendation generation
* `05_evaluation_and_comparison.ipynb` – Performance analysis
* `06_performance_analysis.ipynb` – Final evaluation

---

## Results Summary

| Metric                   | Value       |
| ------------------------ | ----------- |
| Average Similarity Score | 0.58 – 0.69 |
| Genre Alignment          | > 70%       |
| User Coverage            | 100%        |
| Novelty Score            | 0.3 – 0.4   |
| Personalization Variance | 0.02        |

---

## Performance Metrics

* **Fuzzy Jaccard Similarity**
* **Fuzzy Cosine Similarity**
* **Fuzzy Dice Coefficient**

Hybrid similarity formulation:

```
H = 0.4 × Jaccard + 0.4 × Cosine + 0.2 × Dice
```

---

## Technical Implementation Details

### Genre Relationship Graph (Example)

```python
relationships = {
    "Action": {"Adventure": 0.7, "Thriller": 0.6},
    "Comedy": {"Romance": 0.8, "Drama": 0.5},
    "Drama": {"Romance": 0.7}
}
```

### Maximal Marginal Relevance (MMR)

```
MMR = relevance − λ × similarity
```

Used to balance recommendation relevance and diversity.

---

## Dependencies

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
tqdm
requests
```

---

## Project Structure

```
granular_recommendation/
├── src/
├── notebooks/
├── data/
├── results/
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@software{granular_fuzzy_recommendation_2025,
  title = {Granular Fuzzy Recommendation System},
  author = {Nouhaila El Morjani},
  year = {2025},
  note = {Fuzzy logic and granular computing-based recommendation system}
}
```

---
