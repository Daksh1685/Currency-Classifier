import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMAGE_SIZE = 224  # MobileNet default input size
BATCH_SIZE = 32  # Reduced batch size to handle large number of images
EPOCHS = 100  # Increased epochs with early stopping
NUM_CLASSES = 6  # 6 denominations: 10,20,50,100,200,500
INITIAL_LEARNING_RATE = 0.001
MAX_QUEUE_SIZE = 100  # Increased queue size for data loading
WORKERS = 8  # Number of parallel workers for data loading

def create_data_generators():
    """Create train and validation data generators with augmentation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,  # 20% data for validation
        brightness_range=[0.8, 1.2],  # Brightness augmentation
        preprocessing_function=None  # No additional preprocessing to speed up loading
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # First, reorganize the 10 rupee notes to match other folders
    source_dir = Path('.')
    if (source_dir / '10' / '10').exists():
        # Create a temporary directory for training
        temp_dir = source_dir / 'temp_dataset'
        temp_dir.mkdir(exist_ok=True)
        
        # Copy all denomination folders except 10
        for denom in ['20', '50', '100', '200', '500']:
            if (source_dir / denom).exists():
                shutil.copytree(source_dir / denom, temp_dir / denom, dirs_exist_ok=True)
        
        # Copy 10 rupee notes from nested folder
        shutil.copytree(source_dir / '10' / '10', temp_dir / '10', dirs_exist_ok=True)
        
        data_dir = temp_dir
    else:
        data_dir = source_dir

    denominations = ['10', '20', '50', '100', '200', '500']
    class_weights = compute_class_weights(data_dir)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=denominations,
        subset='training',
        shuffle=True
    )

    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=denominations,
        subset='validation',
        shuffle=True
    )

    return train_generator, val_generator, class_weights

def compute_class_weights(data_dir):
    """Compute class weights to handle class imbalance"""
    denominations = ['10', '20', '50', '100', '200', '500']
    class_counts = []
    
    for denom in denominations:
        path = Path(data_dir) / denom
        jpg_count = len(list(path.glob("*.jpg")))
        jpeg_count = len(list(path.glob("*.jpeg")))
        # Count both .jpg and .jpeg files
        count = jpg_count + jpeg_count
        # Ensure we have at least 1 image to avoid division by zero
        class_counts.append(max(1, count))
    
    # Calculate weights
    max_count = max(class_counts)
    class_weights = {i: max_count/count for i, count in enumerate(class_counts)}
    
    # Print class distribution
    print("\nClass distribution:")
    print("-" * 40)
    for denom, count in zip(denominations, class_counts):
        print(f"₹{denom:>3} : {count:>4} images, weight = {class_weights[list(denominations).index(denom)]:.2f}")
    print("-" * 40)
    
    return class_weights

def create_model():
    """Create and compile the MobileNetV2 model"""
    # Load MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Freeze the base model layers initially
    base_model.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def create_callbacks():
    """Create callbacks for training"""
    # Model checkpoint to save best model
    checkpoint = ModelCheckpoint(
        'best_currency_classifier.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Early stopping with increased patience for larger dataset
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,  # Increased patience for larger dataset
        restore_best_weights=True,
        mode='max',
        min_delta=0.001,  # Minimum change to qualify as an improvement
        verbose=1
    )

    # Reduce learning rate when plateau is reached
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,  # More aggressive LR reduction
        patience=8,  # Increased patience
        min_lr=1e-7,  # Lower minimum learning rate
        verbose=1
    )

    return [checkpoint, early_stopping, reduce_lr]

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    """Main training function"""
    print("Creating data generators...")
    train_generator, val_generator, class_weights = create_data_generators()

    print("\nCreating model...")
    model, base_model = create_model()
    callbacks = create_callbacks()

    # First phase: Train only the top layers
    print("\nPhase 1: Training top layers...")
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Second phase: Fine-tune convolutional layers
    print("\nPhase 2: Fine-tuning convolutional layers...")
    # Unfreeze some layers of the base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze all except last 30 layers
        layer.trainable = False

    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continue training
    history2 = model.fit(
        train_generator,
        epochs=EPOCHS-20,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Combine histories
    combined_history = tf.keras.callbacks.History()
    combined_history.history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }

    # Plot training history
    plot_training_history(combined_history)

    return model, combined_history

if __name__ == "__main__":
    print("Starting currency classification training...")
    print("GPU Available:", bool(tf.config.list_physical_devices('GPU')))
    
    # Count total images in dataset
    total_images = 0
    denominations = ['10', '20', '50', '100', '200', '500']
    print("\nDataset Statistics:")
    print("-" * 40)
    for denom in denominations:
        path = Path('.') / denom
        if path.exists():
            count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.jpeg"))) + len(list(path.glob("*.png")))
            total_images += count
            print(f"₹{denom:>3} : {count:>5} images")
    print("-" * 40)
    print(f"Total images: {total_images}")
    print("-" * 40)

    # Train the model
    model, history = train_model()
    
    # Print final metrics
    final_val_accuracy = history.history['val_accuracy'][-1]
    best_val_accuracy = max(history.history['val_accuracy'])
    
    print(f"\nTraining completed!")
    print(f"Final validation accuracy: {final_val_accuracy:.2%}")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print("Training history plot saved as 'training_history.png'")
    print("Best model saved as 'best_currency_classifier.h5'")