from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def create_multilabel_confusion_matrix(y_true, y_pred):
    """
    Calculates and visualises the confusion matrix for each label in 
    a multi-label classification problem.
    
    Args:
        y_true (num_samples, num_classes): True binary labels for each instance.
        y_pred (num_samples, num_classes): Predicted binary labels for each instance.
    """
    # Calculate confusion matrix for multi-label classification
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    # Number of labels (each label has its own confusion matrix)
    num_labels = mcm.shape[0]
    _, axes = plt.subplots(1, num_labels, figsize=(15, 5))

    for i, ax in enumerate(axes):
        # Get confusion matrix for label i
        cm = mcm[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Pred 0", "Pred 1"], 
                    yticklabels=["True 0", "True 1"], ax=ax)
        ax.set_title(f'Label {i+1}')

    plt.tight_layout()
    plt.show()


def print_report(y_true , y_pred):
    """
    Prints and returns the classification report for multi-label classification.
    
    Args:
        y_true (num_samples, num_classes): True binary labels for each instance.
        y_pred (num_samples, num_classes): Predicted binary labels for each instance.
    
    Returns:
        str: The classification report.
    """
    report = classification_report(y_true, y_pred)
    print(report)
    return report
