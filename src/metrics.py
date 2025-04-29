import tensorflow as tf

def hamming_loss(y_true, y_pred, threshold=0.5):
    """
    Computes Hamming Loss for multi-label classification.

    Args:
        y_true (tf.Tensor): Ground truth binary labels (batch_size, num_labels).
        y_pred (tf.Tensor): Predicted probabilities or binary values.
        threshold (float): Threshold to binarize predictions.

    Returns:
        tf.Tensor: Scalar Hamming loss.
    """

    # Binarize predictions using the threshold
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Count mismatches
    mismatches = tf.not_equal(y_true, y_pred_binary)
    mismatches = tf.cast(mismatches, tf.float32)

    # Total elements: batch size * num labels
    total_labels = tf.cast(tf.size(y_true), tf.float32)

    # Hamming loss = average number of misclassified labels
    return tf.reduce_sum(mismatches) / total_labels


def example_based_f1(y_true, y_pred, threshold=0.5):
    """
    Computes example-based F1 score for multi-label classification.

    Args:
        y_true (tf.Tensor): Ground truth binary labels, shape (batch_size, num_labels).
        y_pred (tf.Tensor): Predicted probabilities or binary labels, shape (batch_size, num_labels).
        threshold (float): Threshold to binarize predictions.

    Returns:
        tf.Tensor: Mean example-based F1 score.
    """

    # Binarize predictions
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # True positives per example
    intersection = tf.reduce_sum(y_true * y_pred_binary, axis=1)

    # Precision: TP / (TP + FP)
    predicted_sum = tf.reduce_sum(y_pred_binary, axis=1)
    precision = tf.math.divide_no_nan(intersection, predicted_sum)

    # Recall: TP / (TP + FN)
    true_sum = tf.reduce_sum(y_true, axis=1)
    recall = tf.math.divide_no_nan(intersection, true_sum)

    # F1 score per example
    f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)

    # Average F1 across all examples
    return tf.reduce_mean(f1)



def jaccard_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calculates Jaccard similarity (Intersection over Union) for multi-label classification.

    Args:
        y_true (tf.Tensor): Ground truth binary labels (batch_size, num_classes).
        y_pred (tf.Tensor): Predicted probabilities or binary labels (batch_size, num_classes).
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        tf.Tensor: Mean Jaccard similarity across all samples.
    """

    # Convert predicted probabilities to binary labels using the threshold
    y_pred_binary = tf.cast(y_pred >= threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Compute intersection and union
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred_binary), axis=1)
    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.cast(y_true, tf.bool),
                                                tf.cast(y_pred_binary, tf.bool)), tf.float32), axis=1)

    # Avoid division by zero
    jaccard = tf.math.divide_no_nan(intersection, union)

    # Return mean over the batch
    return tf.reduce_mean(jaccard)