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
    """
    1-NN分类器
    使用选择器选择的原型进行1最近邻分类
    """
    
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
        # 注意：对于大型测试集，这里可以用矩阵运算优化，
        # 但为了代码清晰和内存考虑，循环也是可以接受的（会慢一些）
        for x in X_test:
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            nearest_idx = np.argmin(distances)
            predictions.append(self.prototype_labels[nearest_idx])
        
        return np.array(predictions)
    
    def get_info(self):
        return self.selector.get_info()


# ========================================================================
# Evaluation Functions
# ========================================================================
def evaluate_classifier(classifier, X_test, y_test, verbose=True):
    """
    评估分类器性能
    
    Args:
        classifier: 1-NN分类器实例
        X_test: 测试数据
        y_test: 测试标签
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含准确率、预测时间等指标
    """
    # 预测
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # 计算准确率
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
# Visualization Functions
# ========================================================================
def plot_comparison_results(results_dict, save_path=None):
    """绘制方法对比结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in methods]
    times = [results_dict[m]['prediction_time'] for m in methods]
    
    # 准确率对比
    axes[0].bar(methods, accuracies, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Classification Accuracy Comparison', fontsize=14)
    axes[0].set_ylim([min(accuracies)*0.95, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for i, (method, acc) in enumerate(zip(methods, accuracies)):
        axes[0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=10)
    
    # 预测时间对比
    axes[1].bar(methods, times, color='coral', alpha=0.7)
    axes[1].set_ylabel('Prediction Time (s)', fontsize=12)
    axes[1].set_title('Prediction Time Comparison', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for i, (method, t) in enumerate(zip(methods, times)):
        axes[1].text(i, t + max(times)*0.02, f'{t:.1f}s', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved as {save_path}")
    plt.show()


def plot_prototype_count_analysis(prototype_counts, accuracies, times, save_path=None):
    """绘制原型数量分析"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率 vs 原型数量
    axes[0].plot(prototype_counts, accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[0].set_xlabel('Number of Prototypes per Class', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy vs Prototype Count', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 预测时间 vs 原型数量
    axes[1].plot(prototype_counts, times, marker='s', linewidth=2, markersize=8, color='coral')
    axes[1].set_xlabel('Number of Prototypes per Class', fontsize=12)
    axes[1].set_ylabel('Prediction Time (s)', fontsize=12)
    axes[1].set_title('Prediction Time vs Prototype Count', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrototype count analysis plot saved as {save_path}")
    plt.show()


# ========================================================================
# Experiment Functions
# ========================================================================
def run_comparison_experiment(X_train, X_test, y_train, y_test, num_prototypes=50):
    """
    运行对比实验：Full (Baseline) vs Random vs PCA+KMeans
    """
    print("="*70)
    print("MNIST Prototype Selection - Method Comparison")
    print("="*70)
    
    # 定义选择器
    print(f"\n[1/3] Initializing selectors (prototypes per class: {num_prototypes})...")
    
    selectors = {
        'Baseline': BaselineSelector(),
        'Random_Selection': RandomSelector(
            num_prototypes_per_class=num_prototypes,
            random_state=42
        ),
        'Cluster': ClusterSelector(
            num_prototypes_per_class=num_prototypes,
            random_state=42,
            pca_components=100
        ),
    }
    
    # 训练和评估
    print("\n[2/3] Training and evaluating 1-NN classifiers...")
    results = {}
    
    for name, selector in selectors.items():
        print(f"\n>>> Processing {name}...")
        classifier = OneNNClassifier(selector)
        classifier.fit(X_train, y_train)
        results[name] = evaluate_classifier(classifier, X_test, y_test, verbose=True)
    
    # 可视化结果
    print("\n[3/3] Generating visualizations...")
    os.makedirs('results/figures', exist_ok=True)
    plot_comparison_results(
        results,
        save_path=f'results/figures/method_comparison_{num_prototypes}ppc.png'
    )
    
    # 打印总结
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    baseline_time = results['Baseline']['prediction_time']
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  - Accuracy: {result['accuracy']:.4f}")
        print(f"  - Prediction Time: {result['prediction_time']:.2f}s")
        if name != 'Baseline':
            speedup = baseline_time / result['prediction_time']
            print(f"  - Speedup vs Baseline: {speedup:.2f}x")
    
    return results


def run_parameter_tuning(X_train, X_test, y_train, y_test):
    """
    参数调优实验：测试不同数量的原型
    """
    print("\n" + "="*70)
    print("MNIST Prototype Selection - Parameter Tuning")
    print("="*70)
    
    prototype_counts = [10, 20, 50, 100, 200]
    print(f"\nTesting prototype counts: {prototype_counts}")
    
    accuracies = []
    times = []
    
    for count in prototype_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {count} prototypes per class...")
        print(f"{'='*60}")
        
        # 创建选择器和分类器
        selector = ClusterSelector(
            num_prototypes_per_class=count,
            random_state=42,
            pca_components=100
        )
        classifier = OneNNClassifier(selector)
        
        # 训练和评估
        classifier.fit(X_train, y_train)
        result = evaluate_classifier(classifier, X_test, y_test, verbose=True)
        
        accuracies.append(result['accuracy'])
        times.append(result['prediction_time'])
    
    # 可视化
    os.makedirs('results/figures', exist_ok=True)
    plot_prototype_count_analysis(
        prototype_counts,
        accuracies,
        times,
        save_path='results/figures/parameter_tuning.png'
    )
    
    # 打印最佳结果
    best_idx = np.argmax(accuracies)
    print("\n" + "="*70)
    print("PARAMETER TUNING SUMMARY")
    print("="*70)
    print(f"\nBest accuracy: {accuracies[best_idx]:.4f}")
    print(f"Best prototype count: {prototype_counts[best_idx]} per class")
    print(f"Total prototypes: {prototype_counts[best_idx] * 10}")
    
    return prototype_counts, accuracies, times


# ========================================================================
# Main Function
# ========================================================================
def main():
    """主函数"""
    # 参数解析
    parser = argparse.ArgumentParser(description='MNIST Prototype Selection with 1-NN Classifier')
    parser.add_argument('--experiment', type=str, default='comparison',
                        choices=['comparison', 'tuning', 'both'],
                        help='Experiment type: comparison (default), tuning, or both')
    parser.add_argument('--num_prototypes', type=int, default=50,
                        help='Number of prototypes per class for comparison experiment')
    parser.add_argument('--quick_test', action='store_true',
                        help='Use subset of test data for quick testing')
    
    args = parser.parse_args()
    
    # 数据文件路径
    training_images_filepath = 'data/train-images.idx3-ubyte'
    training_labels_filepath = 'data/train-labels.idx1-ubyte'
    test_images_filepath = 'data/t10k-images.idx3-ubyte'
    test_labels_filepath = 'data/t10k-labels.idx1-ubyte'
    
    # 加载数据
    print("\n" + "="*70)
    print("Loading MNIST Dataset")
    print("="*70)
    
    start_time = time.time()
    
    # 实例化加载器
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
    
    # 快速测试模式
    if args.quick_test:
        print("\n⚠️  Quick test mode: using 1000 test samples")
        X_test = X_test[:1000]
        y_test = y_test[:1000]
    
    # 运行实验
    if args.experiment == 'comparison':
        run_comparison_experiment(X_train, X_test, y_train, y_test, args.num_prototypes)
        
    elif args.experiment == 'tuning':
        run_parameter_tuning(X_train, X_test, y_train, y_test)
        
    elif args.experiment == 'both':
        run_comparison_experiment(X_train, X_test, y_train, y_test, args.num_prototypes)
        run_parameter_tuning(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("✅ All experiments completed successfully!")
    print("📊 Results saved to: ./results/figures/")
    print("="*70)


if __name__ == "__main__":
    main()