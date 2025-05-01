"""
This scripts prints the metrics stored in a pickle file,
in a neat formatted way. Run this script after training,
to ensure pickle file is created.
"""

import pickle


def print_metrics(metrics_file):
    # Load the metrics from the pickle file
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)
    
    print("\n" + "="*50)
    print(f"METRICS REPORT: {metrics_file}")
    print("="*50)
    
    # Print basic info
    print(f"\n{' Best Model Info ':-^50}")
    print(f"Best seed used: {metrics['best_seed']}")
    print(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
    
    # Print average metrics
    print(f"\n{' Average Metrics Across Runs ':-^50}")
    for metric, value in metrics['average_metrics'].items():
        print(f"{metric.replace('_', ' ').title():<15}: {value:.4f}")
    
    # Print final epoch metrics
    print(f"\n{' Final Epoch Training Metrics ':-^50}")
    for metric, value in metrics['final_epoch']['train_metrics'].items():
        print(f"{metric.replace('_', ' ').title():<30}: {value:.4f}")
    
    print(f"\n{' Final Epoch Validation Metrics ':-^50}")
    for metric, value in metrics['final_epoch']['val_metrics'].items():
        metric_name = metric.replace('val_', '').replace('_', ' ').title()
        print(f"{metric_name:<30}: {value:.4f}")
    
    # Print test metrics
    print(f"\n{' Test Set Metrics ':-^50}")
    for metric, value in metrics['test_metrics'].items():
        print(f"{metric.replace('_', ' ').title():<30}: {value:.4f}")
    
    # Print validation metrics (from evaluation)
    print(f"\n{' Validation Set Metrics ':-^50}")
    for metric, value in metrics['val_metrics'].items():
        print(f"{metric.replace('_', ' ').title():<30}: {value:.4f}")
    
    # Print classification report if it exists
    if 'classification_report' in metrics:
        print(f"\n{' Classification Report ':-^50}")
        print(metrics['classification_report'])

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    
    metrics_file = "2d_8_elec_5_grid_size_cnn_metrics.pkl"
    index_to_coordinate = None
    with open("2d_8_elec_5_grid_size_index_to_coordinate.pkl", "rb") as f:
        index_to_coordinate = pickle.load(f)
    
    print_metrics(metrics_file)