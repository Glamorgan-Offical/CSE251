import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data.loader import MnistDataloader 
from selector import BaselineSelector, RandomSelector, ClusterSelector

# ========================================================================
# 1-NN Classifier
# ========================================================================
class OneNNClassifier:
    
    def __init__(self, selector):
        self.selector = selector
        self.prototypes = None
        self.prototype_labels = None
        
    def fit(self, X_train, y_train):
        self.selector.fit(X_train, y_train)
        self.prototypes = self.selector.prototypes
        self.prototype_labels = self.selector.prototype_labels

        return self
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            nearest_idx = np.argmin(distances)
            predictions.append(self.prototype_labels[nearest_idx])
        return np.array(predictions)
    
    def get_info(self):
        return self.selector.get_info()


# ========================================================================
# Evaluation Function
# ========================================================================
def evaluate_classifier(classifier, X_test, y_test, verbose=True):
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    
    results = {
        'accuracy': accuracy,
        'prediction_time': prediction_time,
        'avg_time_per_sample': prediction_time / len(X_test),
        'y_pred': y_pred,
    }
    
    if verbose:
        info = classifier.get_info()
        print(f"\n{'='*60}")
        print(f"Selector: {info['name']}")
        print(f"Num Prototypes: {info['num_prototypes']}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Prediction Time: {prediction_time:.2f}s")
        print(f"Time per sample: {prediction_time/len(X_test)*1000:.2f}ms")
        if info['selection_time']:
            print(f"Selection Time: {info['selection_time']:.2f}s")
        print(f"{'='*60}\n")
    
    return results


# ========================================================================
# Main Function
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description='MNIST Prototype Selection with 1-NN Classifier')
    parser.add_argument('--selector', type=str, required=True,
                        choices=['baseline', 'random', 'cluster'],
                        help='Selector type: baseline, random, or cluster')
    parser.add_argument('--num_prototypes', type=int, default=50,
                        help='Number of prototypes per class (ignored for baseline)')
    parser.add_argument('--pca_components', type=int, default=100,
                        help='PCA components for cluster selector')
    parser.add_argument('--quick_test', action='store_true',
                        help='Use subset of test data for quick testing')
    
    args = parser.parse_args()
    
    training_images_filepath = 'data/train-images.idx3-ubyte'
    training_labels_filepath = 'data/train-labels.idx1-ubyte'
    test_images_filepath = 'data/t10k-images.idx3-ubyte'
    test_labels_filepath = 'data/t10k-labels.idx1-ubyte'
    
    print("\n Loading MNIST Dataset")
    
    start_time = time.time()
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                       test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # 转换为numpy数组并归一化
    X_train = np.array(x_train).reshape(len(x_train), -1) / 255.0
    X_test = np.array(x_test).reshape(len(x_test), -1) / 255.0
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    elapsed_time = time.time() - start_time
    print(f"Data loaded in: {elapsed_time:.2f} seconds")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    
    # 选择器
    print("\n Initializing Selector")
    
    if args.selector == 'baseline':
        selector = BaselineSelector()
        print("Using Baseline (Full Training Set)")
    elif args.selector == 'random':
        selector = RandomSelector(
            num_prototypes_per_class=args.num_prototypes,
            random_state=42
        )
        print(f"Using Random Selection ({args.num_prototypes} prototypes per class)")
    elif args.selector == 'cluster':
        selector = ClusterSelector(
            num_prototypes_per_class=args.num_prototypes,
            random_state=42,
            pca_components=args.pca_components
        )
        print(f"Using Cluster Selection ({args.num_prototypes} prototypes per class, PCA={args.pca_components})")
    
    # 训练和评估
    print("\n Training 1-NN")
    
    classifier = OneNNClassifier(selector)
    classifier.fit(X_train, y_train)
    results = evaluate_classifier(classifier, X_test, y_test, verbose=True)
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_file = f"results/{args.selector}_n{args.num_prototypes}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Selector: {args.selector}\n")
        f.write(f"Num Prototypes per Class: {args.num_prototypes}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Prediction Time: {results['prediction_time']:.2f}s\n")
        f.write(f"Time per Sample: {results['avg_time_per_sample']*1000:.2f}ms\n")
    
    print(f"\nResults saved to: {result_file}")
    print("Experiment completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()