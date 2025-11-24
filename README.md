# CSE 250A Project: Purchase Prediction with HMM

This project implements a Hidden Markov Model (HMM) to predict user purchase intent based on sequential browsing behavior.

## Project Overview

We analyze the [Retailrocket recommender system dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset) to model user sessions. The goal is to classify sessions as either leading to a **purchase** or **no-purchase**.

## Methodology

1.  **Sessionization**: User events are grouped into sessions based on a **30-minute inactivity threshold**. This was determined to be optimal after comparing 10, 20, 30, 60, and 120-minute thresholds.
2.  **Labeling**: Sessions containing a 'transaction' event are labeled as positive (purchase).
3.  **Observation Leakage Prevention**: The 'transaction' event itself is removed from the input sequences to prevent the model from trivially learning the label.
4.  **Data Balancing**: We undersample the majority class (no-purchase) to create a balanced training dataset.
5.  **Model**: We train two separate Categorical HMMs:
    *   $HMM_{\verb|purchase|}$: Trained on purchase sequences.
    *   $HMM_{\verb|no_purchase|}$: Trained on no-purchase sequences.
    *   **Hidden States**: We selected **4 hidden states** (`n_components=4`) after comparing 2, 3, 4, and 5 components. This provided the best balance between model complexity and generalization.
6.  **Inference**: We classify new sequences by comparing the log-likelihood of the sequence under both models.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd cse250a_proj
    ```

2.  **Install Git LFS**:
    This project uses Git LFS for large dataset files.
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Open and run the Jupyter Notebook:

```bash
jupyter notebook Refactor_Analysis.ipynb
```

### Hyperparameter Tuning

To re-run the hyperparameter tuning experiment (Session Thresholds & Number of Components):

```bash
python Hyperparameter_Tuning.py
```

This will generate `hmm_tuning_results.csv` with the accuracy for each combination.

## Results

The model achieves an accuracy of approximately **90%** on the test set.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~0.90 |
| **Precision** | ~0.90 |
| **Recall** | ~0.90 |
| **F1-Score** | ~0.90 |
