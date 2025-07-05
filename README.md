# Nocturne: A Multimedia Recommendation System

Nocturne is a content-based recommendation system that helps users discover movies they may enjoy based on their preferences. Using collaborative filtering with cosine similarity and fuzzy title matching, Nocturne suggests similar movies by analyzing user ratings.

---

## Features

- Fuzzy Search: Intelligent title matching even with spelling mistakes.
- Collaborative Filtering: Recommends content based on user similarity and movie ratings.
- KNN Model with Cosine Similarity: Finds top `n` nearest neighbors for a given movie.
- CSV-based Dataset: Uses `movies.csv` and `ratings.csv` as the input data.

---

## Tech Stack

- Language: Python 3
- Libraries:
  - pandas
  - scikit-learn
  - fuzzywuzzy
- Model: K-Nearest Neighbors (`NearestNeighbors`) with cosine similarity

---

## Sample Input & Output

### Example

**Input:**
Enter a Movie Name of your Taste for Recommendation: toy story

**Output:**
Top 10 Recommended Movies:
1. Toy Story 2 Distance: 0.1003

2. A Bug's Life Distance: 0.1281

3. Monsters, Inc. Distance: 0.1407

4. Finding Nemo Distance: 0.1572

5. The Incredibles Distance: 0.1679

6. Toy Story 3 Distance: 0.1724

7. Shrek Distance: 0.1816

8. Ice Age Distance: 0.1938

9. Despicable Me Distance: 0.2085

10. Cars Distance: 0.2149


