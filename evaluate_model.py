import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
CLASS_NAMES = ['10', '20', '50', '100', '200', '500']

def create_data_generator():
    """Create data generator for evaluation"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # First, reorganize the 10 rupee notes to match other folders
    source_dir = Path('.')
    if (source_dir / '10' / '10').exists():
        # Create a temporary directory for evaluation
        temp_dir = source_dir / 'temp_dataset'
        temp_dir.mkdir(exist_ok=True)
        
        # Copy all denomination folders except 10
        for denom in CLASS_NAMES:
            if denom == '10':
                if not (temp_dir / '10').exists():
                    temp_dir.mkdir(exist_ok=True)
                    import shutil
                    shutil.copytree(source_dir / '10' / '10', temp_dir / '10', dirs_exist_ok=True)
            elif (source_dir / denom).exists():
                if not (temp_dir / denom).exists():
                    import shutil
                    shutil.copytree(source_dir / denom, temp_dir / denom, dirs_exist_ok=True)
        
        data_dir = temp_dir
    else:
        data_dir = source_dir

    # Create generator
    generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False  # Important for matching predictions with filenames
    )
    
    return generator

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def calculate_metrics(cm, class_names):
    """Calculate precision, recall, and F1 score for each class"""
    metrics = {}
    n_classes = len(class_names)
    
    for i in range(n_classes):
        # True Positives
        tp = cm[i, i]
        # False Positives
        fp = np.sum(cm[:, i]) - tp
        # False Negatives
        fn = np.sum(cm[i, :]) - tp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_names[i]] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    # Calculate macro averages
    macro_precision = np.mean([m['Precision'] for m in metrics.values()])
    macro_recall = np.mean([m['Recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['F1-Score'] for m in metrics.values()])
    
    metrics['Macro Average'] = {
        'Precision': macro_precision,
        'Recall': macro_recall,
        'F1-Score': macro_f1
    }
    
    return metrics

def evaluate_model():
    """Evaluate model performance"""
    print("Loading model...")
    model = load_model('best_currency_classifier.h5')
    
    print("Creating data generator...")
    generator = create_data_generator()
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = generator.classes
    
    # Calculate metrics
    print("\nCalculating metrics...")
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=[f"₹{name}" for name in CLASS_NAMES],
        digits=4
    )
    
    # Create and plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(true_classes, predicted_classes)
    plot_confusion_matrix(cm, [f"₹{name}" for name in CLASS_NAMES])
    
    # Calculate detailed metrics
    print("\nCalculating detailed metrics...")
    metrics = calculate_metrics(cm, [f"₹{name}" for name in CLASS_NAMES])
    
    # Print results
    print("\nDetailed Performance Metrics:")
    print("=" * 80)
    print(report)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")
    
    # Print detailed metrics
    print("\nDetailed Metrics by Class:")
    print("=" * 80)
    for class_name, class_metrics in metrics.items():
        print(f"\n{class_name}:")
        print(f"Precision: {class_metrics['Precision']:.4f}")
        print(f"Recall   : {class_metrics['Recall']:.4f}")
        print(f"F1-Score : {class_metrics['F1-Score']:.4f}")
    
    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 40)
    for i, class_name in enumerate(CLASS_NAMES):
        class_indices = true_classes == i
        class_accuracy = np.mean(predicted_classes[class_indices] == true_classes[class_indices])
        print(f"₹{class_name:>3} : {class_accuracy:.2%}")

if __name__ == "__main__":
    evaluate_model()