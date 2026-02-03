import os
import argparse
import time
import numpy as np
import scipy.stats as stats
from sklearn.metrics import accuracy_score

from data.loader import MnistDataloader 
from selector import BaselineSelector, RandomSelector, ClusterSelector

# 1-NN Classifier
class OneNNClassifier:
    
    def __init__(self, selector):
        self.selector = selector
        self.prototypes = None
        self.labels = None
        
    def fit(self, X_train, y_train):
        self.selector.fit(X_train, y_train)
        self.prototypes = self.selector.prototypes
        self.labels = self.selector.labels

        return self
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            nearest_idx = np.argmin(distances)
            predictions.append(self.labels[nearest_idx])
        return np.array(predictions)
    
    def get_info(self):
        return self.selector.get_info()


def train(X_train, y_train, X_test, y_test, args, seed):
    
    # Choose Selector
    if args.selector == 'baseline':
        selector = BaselineSelector()
    elif args.selector == 'random':
        selector = RandomSelector(
            num_prototypes_per_class=args.num_prototypes,
            random_state=seed
        )
    elif args.selector == 'cluster':
        selector = ClusterSelector(
            num_prototypes_per_class=args.num_prototypes,
            random_state=seed,
            pca_components=args.pca_components
        )
    
    # Training
    start_train = time.time()
    classifier = OneNNClassifier(selector)
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    y_pred = classifier.predict(X_test)
    pred_time = time.time() - start_pred
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, train_time, pred_time

# Main Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selector', type=str, required=True,
                         choices=['baseline', 'random', 'cluster'])
    parser.add_argument('--num_prototypes', type=int, default=50,
                         help='Prototypes Per Class')
    parser.add_argument('--pca_components', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=1,
                         help='Number of repetitions')
    
    args = parser.parse_args()
    
    # Dataset path
    training_images_filepath = 'data/train-images.idx3-ubyte'
    training_labels_filepath = 'data/train-labels.idx1-ubyte'
    test_images_filepath = 'data/t10k-images.idx3-ubyte'
    test_labels_filepath = 'data/t10k-labels.idx1-ubyte'
    
    print("\n Loading MNIST Dataset")
    
    start_time = time.time()
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # Normalization
    X_train = np.array(x_train).reshape(len(x_train), -1) / 255.0
    X_test = np.array(x_test).reshape(len(x_test), -1) / 255.0
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if args.selector == 'baseline':
        args.n_runs = 1

    accuracies = []
    train_times = []
    pred_times = []
    
    print(f"Running {args.n_runs} experiments (M={args.num_prototypes*10} total)...")
    
    for i in range(args.n_runs):
        seed = 114514 + i
        print(f"  Run {i+1}/{args.n_runs} (seed={seed})...", end="", flush=True)
        
        acc, t_train, t_pred = train(X_train, y_train, X_test, y_test, args, seed)
        
        accuracies.append(acc)
        train_times.append(t_train)
        pred_times.append(t_pred)
        print(f" Done. Acc={acc:.4f}")

    mean_acc = np.mean(accuracies)
    mean_time = np.mean(pred_times)
    
    # Calculate Confidence Interval
    if args.n_runs > 1:
        se = np.std(accuracies, ddof=1) / np.sqrt(args.n_runs)
        # t-distribution critical value for 95% CI
        t_crit = stats.t.ppf(0.975, df=args.n_runs-1)
        ci_95 = t_crit * se
    else:
        ci_95 = 0.0

    # Save
    os.makedirs('results_final', exist_ok=True)
    if args.selector == 'cluster':
        fname = f"results_final/{args.selector}_n{args.num_prototypes}_pca{args.pca_components}.txt"
    else:
        fname = f"results_final/{args.selector}_n{args.num_prototypes}.txt"
        
    with open(fname, 'w') as f:
        f.write(f"Selector: {args.selector}\n")
        f.write(f"Prototypes per Class: {args.num_prototypes}\n")
        f.write(f"Total Prototypes (M): {args.num_prototypes * 10}\n")
        f.write(f"Runs: {args.n_runs}\n")
        f.write(f"Mean Accuracy: {mean_acc:.5f}\n")
        f.write(f"95% CI: {ci_95:.5f}\n")
        f.write(f"Mean Prediction Time: {mean_time:.4f}s\n")
        f.write(f"Raw Accuracies: {accuracies}\n")
    
    print(f"\nFinal Result (Average over {args.n_runs} runs):")
    print(f"Accuracy: {mean_acc:.4f} ± {ci_95:.4f}")
    print(f"Time: {mean_time:.2f}s")

if __name__ == "__main__":
    main()