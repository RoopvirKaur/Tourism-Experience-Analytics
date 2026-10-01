"""
Recommendation Engine Module for Tourism Experience Analytics.
Implements Content-Based Filtering (TF-IDF + Cosine Similarity), Collaborative Filtering (User-Item Matrix + User Similarity),
and a Hybrid Recommendation System with cold-start handling and offline evaluation.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessor import build_consolidated_dataset
from src.features.build_features import build_all_features


class ContentBasedRecommender:
    """
    Content-Based Recommendation Engine using TF-IDF feature matrix of attractions.
    """

    def __init__(self, tfidf_matrix: np.ndarray, items_df: pd.DataFrame):
        self.tfidf_matrix = tfidf_matrix
        self.items_df = items_df.copy().reset_index(drop=True)
        self.item_ids = list(self.items_df["AttractionId"])
        self.item_id_to_idx = {att_id: idx for idx, att_id in enumerate(self.item_ids)}
        self.item_sim_matrix = cosine_similarity(self.tfidf_matrix)

    def predict_scores_for_user(self, user_visited_ids: list, user_ratings: list = None) -> np.ndarray:
        """
        Generates content-based preference scores for all attractions given a user's visited item IDs.
        """
        valid_indices = [self.item_id_to_idx[aid] for aid in user_visited_ids if aid in self.item_id_to_idx]
        if not valid_indices:
            return np.zeros(len(self.items_df))

        if user_ratings is not None and len(user_ratings) == len(valid_indices):
            weights = np.array(user_ratings, dtype=float)
            w_sum = weights.sum()
            weights = weights / w_sum if w_sum > 0 else np.ones(len(valid_indices)) / len(valid_indices)
            user_profile = (self.tfidf_matrix[valid_indices].T @ weights)
        else:
            user_profile = self.tfidf_matrix[valid_indices].mean(axis=0)

        scores = cosine_similarity(user_profile.reshape(1, -1), self.tfidf_matrix).flatten()
        return scores


class CollaborativeFilteringRecommender:
    """
    User-User Collaborative Filtering Engine on sparse User-Item rating pivot matrix.
    Computes single-user similarity vectors on demand for memory and runtime efficiency.
    """

    def __init__(self, user_item_matrix: pd.DataFrame):
        self.user_item_matrix = user_item_matrix
        self.user_ids = list(user_item_matrix.index)
        self.item_ids = list(user_item_matrix.columns)
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.ratings_array = user_item_matrix.values

    def predict_scores_for_user(self, user_id: int) -> np.ndarray:
        """
        Generates collaborative filtering rating predictions across all items for a given UserId.
        """
        if user_id not in self.user_id_to_idx:
            return np.zeros(len(self.item_ids))

        u_idx = self.user_id_to_idx[user_id]
        target_vector = self.ratings_array[u_idx:u_idx + 1]
        user_sims = cosine_similarity(target_vector, self.ratings_array).flatten()
        user_sims[u_idx] = 0.0  # Exclude self-similarity

        sim_sum = np.abs(user_sims).sum()
        if sim_sum == 0:
            return np.zeros(len(self.item_ids))

        scores = (user_sims @ self.ratings_array) / (sim_sum + 1e-9)
        return scores


class HybridRecommender:
    """
    Hybrid Recommender combining Content-Based and Collaborative Filtering scores.
    Handles cold-start users and prevents recommending previously visited attractions.
    """

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        cf_recommender: CollaborativeFilteringRecommender,
        master_df: pd.DataFrame,
        items_df: pd.DataFrame,
        alpha: float = 0.5
    ):
        self.content_recommender = content_recommender
        self.cf_recommender = cf_recommender
        self.items_df = items_df.copy().reset_index(drop=True)
        self.master_df = master_df.copy()
        self.alpha = alpha

        # User historical interactions
        self.user_history = {}
        for uid, group in master_df.groupby("UserId"):
            self.user_history[uid] = list(zip(group["AttractionId"], group["Rating"]))

        # Build popularity fallback table for cold-start handling
        item_stats = master_df.groupby("AttractionId").agg(
            Attraction_Avg_Rating=("Rating", "mean"),
            Review_Count=("Rating", "count")
        ).reset_index()

        pop_items = items_df.merge(item_stats, on="AttractionId", how="left")
        pop_items["Attraction_Avg_Rating"] = pop_items["Attraction_Avg_Rating"].fillna(master_df["Rating"].mean())
        pop_items["Review_Count"] = pop_items["Review_Count"].fillna(0)
        pop_items["Popularity_Score"] = (
            0.7 * pop_items["Attraction_Avg_Rating"] + 0.3 * np.log1p(pop_items["Review_Count"])
        )
        self.popular_items = pop_items.sort_values(by="Popularity_Score", ascending=False)

    def recommend(self, user_id: int = None, top_k: int = 5, alpha: float = None) -> pd.DataFrame:
        """
        Generates Top-K recommended attractions for a specified UserId or cold-start fallback.

        Args:
            user_id: Target UserId (can be None or unknown ID for cold-start).
            top_k: Number of recommendations to return.
            alpha: Score weight balance (0.0 = Content-Only, 1.0 = CF-Only).

        Returns:
            pd.DataFrame: Ranked dataframe of top attractions with metadata and scores.
        """
        if alpha is None:
            alpha = self.alpha

        user_history = self.user_history.get(user_id, [])
        visited_ids = set(aid for aid, r in user_history)

        # Cold-Start / Unknown User handling
        if not user_id or user_id not in self.user_history or not user_history:
            candidate_pool = self.popular_items[~self.popular_items["AttractionId"].isin(visited_ids)].copy()
            candidate_pool = candidate_pool.sort_values(by="Popularity_Score", ascending=False)

            selected_rows = []
            country_counts = {}
            penalty = 0.5  # Deterministic geographic diversity penalty

            pool_records = candidate_pool.to_dict('records')

            while len(selected_rows) < top_k and pool_records:
                best_idx = -1
                best_adj_score = -float('inf')

                for idx, item in enumerate(pool_records):
                    cid = item.get("AttractionCountryId", 0)
                    cnt = country_counts.get(cid, 0)
                    adj_score = item["Popularity_Score"] - (penalty * cnt)
                    if adj_score > best_adj_score:
                        best_adj_score = adj_score
                        best_idx = idx

                if best_idx >= 0:
                    chosen = pool_records.pop(best_idx)
                    selected_rows.append(chosen)
                    cid = chosen.get("AttractionCountryId", 0)
                    country_counts[cid] = country_counts.get(cid, 0) + 1
                else:
                    break

            recs = pd.DataFrame(selected_rows)
            recs["Score"] = recs["Popularity_Score"]
            recs["Recommendation_Type"] = "Popularity Fallback (Cold-Start Diversity)"
            output_cols = [
                "AttractionId", "Attraction", "AttractionType",
                "AttractionCityName", "AttractionCountryId", "AttractionAddress", "Attraction_Avg_Rating",
                "Score", "Recommendation_Type"
            ]
            existing_cols = [c for c in output_cols if c in recs.columns]
            return recs[existing_cols].reset_index(drop=True)

        visited_aids = [aid for aid, r in user_history]
        visited_ratings = [r for aid, r in user_history]

        # 1. Content-based scores
        cb_raw = self.content_recommender.predict_scores_for_user(visited_aids, visited_ratings)
        cb_min, cb_max = cb_raw.min(), cb_raw.max()
        cb_norm = (cb_raw - cb_min) / (cb_max - cb_min + 1e-9) if cb_max > cb_min else cb_raw

        # Map CB scores to items_df by AttractionId
        cb_dict = dict(zip(self.content_recommender.items_df["AttractionId"], cb_norm))

        # 2. Collaborative Filtering scores
        cf_raw = self.cf_recommender.predict_scores_for_user(user_id)
        cf_min, cf_max = cf_raw.min(), cf_raw.max()
        cf_norm = (cf_raw - cf_min) / (cf_max - cf_min + 1e-9) if cf_max > cf_min else cf_raw
        cf_dict = dict(zip(self.cf_recommender.item_ids, cf_norm))

        # Combine scores
        results = self.items_df.copy()
        cb_vector = results["AttractionId"].map(cb_dict).fillna(0.0).values
        cf_vector = results["AttractionId"].map(cf_dict).fillna(0.0).values

        hybrid_vector = alpha * cf_vector + (1.0 - alpha) * cb_vector
        results["Score"] = hybrid_vector
        results["Recommendation_Type"] = "Hybrid (CF + Content)"

        # Exclude previously visited/rated attractions
        results = results[~results["AttractionId"].isin(visited_ids)]

        # Attach item average rating
        item_means = self.master_df.groupby("AttractionId")["Rating"].mean().to_dict()
        results["Attraction_Avg_Rating"] = results["AttractionId"].map(item_means).fillna(3.5)

        # Sort and select top K
        results = results.sort_values(by="Score", ascending=False).head(top_k)

        output_cols = [
            "AttractionId", "Attraction", "AttractionType",
            "AttractionCityName", "AttractionCountryId", "AttractionAddress", "Attraction_Avg_Rating",
            "Score", "Recommendation_Type"
        ]
        existing_cols = [c for c in output_cols if c in results.columns]
        return results[existing_cols].reset_index(drop=True)


def evaluate_recommender_offline(recommender: HybridRecommender, master_df: pd.DataFrame, top_k: int = 5, n_test_users: int = 500) -> dict:
    """
    Performs offline evaluation of the recommendation engine using Precision@K, Recall@K, and MAP@K.
    Hold-out evaluation on active users with multiple ratings.
    """
    user_counts = master_df.groupby("UserId")["AttractionId"].count()
    eligible_users = user_counts[user_counts >= 3].index.tolist()

    np.random.seed(42)
    sample_users = np.random.choice(eligible_users, size=min(n_test_users, len(eligible_users)), replace=False)

    # Pre-group user ratings to eliminate repeated DataFrame filtering inside evaluation loop
    user_tx_dict = {}
    for uid, grp in master_df.groupby("UserId"):
        user_tx_dict[uid] = grp.sort_values(by="Rating", ascending=False)["AttractionId"].tolist()

    precisions = []
    recalls = []
    ap_scores = []

    for uid in sample_users:
        all_rated = user_tx_dict[uid]
        # Hold out the highest rated attraction for evaluation
        test_item = all_rated[0]
        train_items = all_rated[1:]

        # Get recommendations using only train items
        cb_scores = recommender.content_recommender.predict_scores_for_user(train_items)
        cb_min, cb_max = cb_scores.min(), cb_scores.max()
        cb_norm = (cb_scores - cb_min) / (cb_max - cb_min + 1e-9) if cb_max > cb_min else cb_scores
        cb_dict = dict(zip(recommender.content_recommender.items_df["AttractionId"], cb_norm))

        cf_scores = recommender.cf_recommender.predict_scores_for_user(uid)
        cf_min, cf_max = cf_scores.min(), cf_scores.max()
        cf_norm = (cf_scores - cf_min) / (cf_max - cf_min + 1e-9) if cf_max > cf_min else cf_scores
        cf_dict = dict(zip(recommender.cf_recommender.item_ids, cf_norm))

        res_df = recommender.items_df.copy()
        cb_v = res_df["AttractionId"].map(cb_dict).fillna(0.0).values
        cf_v = res_df["AttractionId"].map(cf_dict).fillna(0.0).values

        res_df["Score"] = 0.5 * cf_v + 0.5 * cb_v
        # Filter out train items
        res_df = res_df[~res_df["AttractionId"].isin(train_items)]
        top_recs = res_df.sort_values(by="Score", ascending=False).head(top_k)["AttractionId"].tolist()

        hit = 1 if test_item in top_recs else 0
        precisions.append(hit / top_k)
        recalls.append(hit / 1.0)

        # Average Precision at K
        if hit:
            rank = top_recs.index(test_item) + 1
            ap_scores.append(1.0 / rank)
        else:
            ap_scores.append(0.0)

    metrics = {
        "Precision@K": float(np.mean(precisions)),
        "Recall@K": float(np.mean(recalls)),
        "MAP@K": float(np.mean(ap_scores)),
        "Evaluated_Users": len(sample_users),
        "K": top_k
    }

    return metrics


def train_and_evaluate_recommender(artifacts_dir: Path = None):
    """
    Main workflow to build, evaluate, and serialize recommendation engine.
    """
    if artifacts_dir is None:
        artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Building feature matrices and profiles for recommender...")
    feature_data = build_all_features()
    master_df = feature_data["master_df"]
    items_df = feature_data["items_df"]
    tfidf_matrix = feature_data["tfidf_matrix"]
    user_item_matrix = feature_data["user_item_matrix"]

    print("Initializing Content-Based Recommender...")
    cb_rec = ContentBasedRecommender(tfidf_matrix, items_df)

    print("Initializing Collaborative Filtering Recommender...")
    cf_rec = CollaborativeFilteringRecommender(user_item_matrix)

    print("Initializing Hybrid Recommender...")
    hybrid_rec = HybridRecommender(cb_rec, cf_rec, master_df, items_df, alpha=0.5)

    print("\nRunning offline recommendation evaluation...")
    eval_metrics = evaluate_recommender_offline(hybrid_rec, master_df, top_k=5, n_test_users=500)
    print(f"Offline Recommendation Metrics (K={eval_metrics['K']}):")
    print(f" - MAP@{eval_metrics['K']}: {eval_metrics['MAP@K']:.4f}")
    print(f" - Precision@{eval_metrics['K']}: {eval_metrics['Precision@K']:.4f}")
    print(f" - Recall@{eval_metrics['K']}: {eval_metrics['Recall@K']:.4f}")

    # Save artifact
    artifact_bundle = {
        "hybrid_recommender": hybrid_rec,
        "content_recommender": cb_rec,
        "cf_recommender": cf_rec,
        "items_df": items_df,
        "evaluation_metrics": eval_metrics
    }

    recommender_path = artifacts_dir / "recommender.pkl"
    joblib.dump(artifact_bundle, recommender_path)
    print(f"\nSaved recommendation system artifact to {recommender_path}")

    return artifact_bundle, eval_metrics


if __name__ == "__main__":
    train_and_evaluate_recommender()
