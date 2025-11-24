import pandas as pd
from hmmlearn import hmm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import itertools

def run_experiment(threshold_minutes, n_components):
    print(f"\n--- Running Experiment: Threshold={threshold_minutes}m, n_components={n_components} ---")
    
    # Load data
    df = pd.read_csv('dataset/events.csv')
    df = df.sort_values(by=['visitorid', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Sessionization
    df['time_diff'] = df.groupby('visitorid')['timestamp'].diff()
    session_threshold = pd.Timedelta(minutes=threshold_minutes)
    df['new_session'] = (df['time_diff'].isna()) | (df['time_diff'] > session_threshold)
    df['session_id'] = df.groupby('visitorid')['new_session'].cumsum()
    df['unique_session_id'] = df['visitorid'].astype(str) + '_' + df['session_id'].astype(str)

    # Labeling
    session_labels = df.groupby('unique_session_id')['event'].apply(lambda x: 1 if 'transaction' in x.values else 0)

    # Filtering
    df_filtered = df[df['event'] != 'transaction'].copy()
    sequences = df_filtered.groupby('unique_session_id')['event'].apply(list)
    sequences = sequences[sequences.apply(len) >= 2]
    labels = session_labels[sequences.index]

    # Balancing
    data = pd.DataFrame({'sequence': sequences, 'label': labels})
    data_majority = data[data.label == 0]
    data_minority = data[data.label == 1]
    
    if len(data_minority) == 0:
        print("Error: No purchase sequences found!")
        return 0

    data_majority_downsampled = resample(data_majority, 
                                         replace=False,
                                         n_samples=len(data_minority),
                                         random_state=42) 
    data_balanced = pd.concat([data_majority_downsampled, data_minority])

    # Training
    observations = {'view': 0, 'addtocart': 1}
    
    def train_hmm(sequences, n_comp):
        train_seq = []
        for seq in sequences:
            encoded_seq = [observations[obv] for obv in seq if obv in observations]
            if encoded_seq:
                train_seq.append(encoded_seq)
        
        if not train_seq:
            return None

        train_seq_fixed = np.concatenate(train_seq).reshape(-1, 1)
        lengths = [len(seq) for seq in train_seq]
        
        model = hmm.CategoricalHMM(n_components=n_comp, n_iter=100, random_state=42)
        model.fit(train_seq_fixed, lengths)
        return model

    X_train, X_test, y_train, y_test = train_test_split(data_balanced['sequence'], data_balanced['label'], test_size=0.2, random_state=42)

    hmm_purchase = train_hmm(X_train[y_train == 1], n_components)
    hmm_no_purchase = train_hmm(X_train[y_train == 0], n_components)

    # Evaluation
    correct = 0
    total = 0

    for seq, true_label in zip(X_test, y_test):
        encoded_seq = [observations[obv] for obv in seq if obv in observations]
        if not encoded_seq:
            continue
            
        encoded_seq = np.array(encoded_seq).reshape(-1, 1)
        
        try:
            score_purchase = hmm_purchase.score(encoded_seq)
        except:
            score_purchase = -np.inf
            
        try:
            score_no_purchase = hmm_no_purchase.score(encoded_seq)
        except:
            score_no_purchase = -np.inf
            
        pred_label = 1 if score_purchase > score_no_purchase else 0
        
        if pred_label == true_label:
            correct += 1
        total += 1

    acc = correct/total
    print(f"Accuracy: {acc:.4f}")
    return acc

thresholds = [10, 20, 30, 60, 120]
n_components_list = [2, 3, 4, 5]

experiment_results = []

print("Starting Hyperparameter Tuning...")

for t, n in itertools.product(thresholds, n_components_list):
    acc = run_experiment(t, n)

    experiment_results.append({
        'threshold_minutes': t,
        'n_components': n,
        'accuracy': acc
    })

results_df = pd.DataFrame(experiment_results)

output_filename = 'hmm_tuning_results.csv'
results_df.to_csv(output_filename, index=False)

print(f"\n--- Processing Complete ---")
print(f"Results saved to: {output_filename}")

best_row = results_df.loc[results_df['accuracy'].idxmax()]
print(f"\nBest Parameters found: Threshold={int(best_row['threshold_minutes'])} mins, n_components={int(best_row['n_components'])}")
print(f"Best Accuracy: {best_row['accuracy']:.4f}")
