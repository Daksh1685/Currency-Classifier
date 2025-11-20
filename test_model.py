import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class CurrencyPredictor:
    def __init__(self, model_path='best_currency_classifier.h5'):
        self.model = load_model(model_path)
        self.image_size = 224
        self.class_names = ['10', '20', '50', '100', '200', '500']
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        # Load and resize image
        img = load_img(image_path, target_size=(self.image_size, self.image_size))
        
        # Convert to array and add batch dimension
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize
        img_array = img_array / 255.0
        
        return img_array
    
    def predict(self, image_path):
        """Predict denomination of currency note"""
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Get prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return {
            'denomination': self.class_names[predicted_class],
            'confidence': confidence,
            'predictions': dict(zip(self.class_names, predictions[0].tolist()))
        }

def test_on_images(image_paths):
    """Test model on multiple images"""
    predictor = CurrencyPredictor()
    
    print("\nTesting images...")
    print("-" * 60)
    
    for image_path in image_paths:
        try:
            result = predictor.predict(image_path)
            
            print(f"\nImage: {Path(image_path).name}")
            print(f"Predicted Denomination: ₹{result['denomination']}")
            print(f"Confidence: {result['confidence']:.2%}")
            
            # Show top 3 predictions
            print("\nTop 3 predictions:")
            sorted_preds = sorted(result['predictions'].items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:3]
            for denom, conf in sorted_preds:
                print(f"₹{denom:>3} : {conf:.2%}")
            
        except Exception as e:
            print(f"\nError processing {Path(image_path).name}: {str(e)}")
        
        print("-" * 60)

def main():
    """Main function to demonstrate usage"""
    print("Currency Note Classifier")
    print("=" * 60)
    
    # Test on some example images from each denomination
    test_images = []
    
    # Get one random image from each denomination folder
    for denom in ['10', '20', '50', '100', '200', '500']:
        if denom == '10':
            denom_path = Path('.') / '10' / '10'
        else:
            denom_path = Path('.') / denom
            
        if denom_path.exists():
            images = list(denom_path.glob('*.jpg'))
            if images:
                test_images.append(str(np.random.choice(images)))
    
    if test_images:
        test_on_images(test_images)
    else:
        print("No test images found!")

if __name__ == "__main__":
    main()