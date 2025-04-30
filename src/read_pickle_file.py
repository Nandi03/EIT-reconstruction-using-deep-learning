"""
This scripts prints the metrics stored in a pickle file,
in a neat formatted way. Run this script after training,
to ensure pickle file is created.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

def create_cylindrical_f1_heatmap(classification_report_str, index_to_coordinate):
    """
    Creates a 3D surface plot with:
    - Exact coordinates from index_to_coordinate
    - F1-scores from classification report
    - Proper linear interpolation in 3D space
    """
    # Parse F1-scores
    f1_scores = {}
    for line in classification_report_str.split('\n'):
        if not line.strip() or 'avg' in line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            f1_scores[int(parts[0])] = float(parts[3])

    # Prepare data
    points = np.array([coord for idx, coord in index_to_coordinate.items() if idx in f1_scores])
    values = np.array([f1_scores[idx] for idx in index_to_coordinate if idx in f1_scores])

    # Create grid for interpolation
    grid_x, grid_y, grid_z = np.mgrid[
        min(points[:,0]):max(points[:,0]):20j,
        min(points[:,1]):max(points[:,1]):20j,
        min(points[:,2]):max(points[:,2]):20j
    ]

    # Interpolate values
    grid_values = griddata(
        points, values,
        (grid_x, grid_y, grid_z),
        method='linear'
    )

    # Create plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot interpolated values
    ax.voxels(
        ~np.isnan(grid_values),  
        facecolors=plt.cm.viridis(Normalize(vmin=0, vmax=1)(grid_values)),
        edgecolor='none', alpha=0.3
    )
    
    # Plot original points
    sc = ax.scatter(
        points[:,0], points[:,1], points[:,2],
        c=values, cmap='viridis', edgecolor='black',
        s=100, alpha=1, vmin=0, vmax=1
    )

    # Plot Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('F1-Score')

    ax.set_title('3D F1-Score Interpolation')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

    return fig

def create_circular_f1_heatmap(classification_report_str, index_to_coordinate, num_electrodes, sample_points):
    # Parse the classification report
    f1_scores = {}
    lines = classification_report_str.split('\n')
    for line in lines:
        if not line.strip() or 'avg' in line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            class_idx = int(parts[0])
            f1_score = float(parts[3])
            f1_scores[class_idx] = f1_score

    # Prepare electrode positions
    electrode_angles = np.linspace(0, 2*np.pi, num_electrodes, endpoint=False)
    electrode_positions = np.column_stack([np.cos(electrode_angles), np.sin(electrode_angles)])

    # Map F1-scores to electrode positions
    points = []
    values = []
    for idx, coord in index_to_coordinate.items():
        if idx in f1_scores:
            points.append(coord)
            values.append(f1_scores[idx])

    points = np.array(points)
    values = np.array(values)

    # Create grid for interpolation
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 50)
    T, R = np.meshgrid(theta, r)
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Interpolate using nearest-neighbor for missing values
    grid_z = griddata(points, values, (X, Y), method='cubic')
    if np.isnan(grid_z).any():
        grid_z_nn = griddata(points, values, (X, Y), method='nearest')
        grid_z = np.where(np.isnan(grid_z), grid_z_nn, grid_z)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    norm = Normalize(vmin=0, vmax=1)
    im = ax.pcolormesh(T, R, grid_z, norm=norm, cmap='viridis', shading='auto')

    # Plot Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label('F1-Score', rotation=270, labelpad=25, fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    # Plot electrodes
    for i, (angle, pos) in enumerate(zip(electrode_angles, electrode_positions)):
        ax.scatter(angle, 1, c='red', s=100, edgecolor='white', zorder=10)
        ax.text(angle, 1.05, f'E{i+1}', ha='center', va='center', color='white',
                bbox=dict(facecolor='red', alpha=0.7, boxstyle='round'), fontsize=12)

    ax.set_title(f'F1-Score Heatmap for the 2D circular skin \n with {num_electrodes} Electrodes, {sample_points} sample points, using 1D-CNN', pad=20, fontsize=20)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_position([0.05, 0.1, 0.7, 0.8])
    
    plt.tight_layout()
    return fig



def print_metrics(metrics_file, index_to_coordinate, num_electrodes, save_path, sample_points):
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

    fig = create_circular_f1_heatmap(metrics['classification_report'], index_to_coordinate, num_electrodes, sample_points)
    # # fig = create_cylindrical_f1_heatmap(metrics['classification_report'], index_to_coordinate)
    fig.savefig(save_path)
    # plt.show()
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    
    metrics_file = "2d_8_elec_5_grid_size_cnn_metrics.pkl"
    index_to_coordinate = None
    with open("2d_8_elec_5_grid_size_index_to_coordinate.pkl", "rb") as f:
        index_to_coordinate = pickle.load(f)
    
    print_metrics(metrics_file, index_to_coordinate, 16, "2d_8_elec_5_grid_size_cnn_f1_scores.png", 26)