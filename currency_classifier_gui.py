import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import threading
from gtts import gTTS
import pygame

class CurrencyClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Currency Classifier")
        self.root.geometry("700x750")
        
        # Initialize audio with higher volume
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1.0)
        
        # Configure style for a more minimal look
        style = ttk.Style()
        style.configure('Browse.TButton', font=('Segoe UI', 10))
        style.configure('TLabelframe', borderwidth=1)  # Thinner borders
        style.configure('TLabelframe.Label', font=('Segoe UI', 10), padding=5)
        style.configure('Result.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Info.TLabel', font=('Segoe UI', 9))
        style.configure('Clean.TRadiobutton', font=('Segoe UI', 9))
        style.configure('TProgressbar', thickness=6)  # Thinner progress bar
        
        # Configure Treeview for a cleaner look
        style.configure('Treeview', rowheight=25, font=('Segoe UI', 9))
        style.configure('Treeview.Heading', font=('Segoe UI', 9, 'bold'))
        
        # Background color
        self.root.configure(bg='#ffffff')
        
        # Hindi number mapping
        self.hindi_numbers = {
            '10': 'दस',
            '20': 'बीस',
            '50': 'पचास',
            '100': 'सौ',
            '200': 'दो सौ',
            '500': 'पाँच सौ'
        }
        
        # Language preference (default to English)
        self.language = tk.StringVar(value="English")
        
        # Load the model
        self.load_model()
        
        # Create GUI elements
        self.create_widgets()
        
    def load_model(self):
        """Load the trained model"""
        self.model = load_model('best_currency_classifier.h5')
        self.image_size = 224
        self.class_names = ['10', '20', '50', '100', '200', '500']
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame with minimal padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Language selection section
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Language selection - minimal design
        lang_label = ttk.Label(top_frame, text="Language:", style='Info.TLabel')
        lang_label.grid(row=0, column=0, padx=(0, 10))
        
        # Language radio buttons
        ttk.Radiobutton(top_frame, text="English", variable=self.language,
                       value="English", style='Clean.TRadiobutton').grid(row=0, column=1, padx=10)
        ttk.Radiobutton(top_frame, text="हिंदी", variable=self.language,
                       value="Hindi", style='Clean.TRadiobutton').grid(row=0, column=2, padx=10)
        
        # Image section
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Create a frame for the image with subtle border
        image_container = ttk.Frame(image_frame, borderwidth=1, relief="solid", width=350, height=250)
        image_container.grid(row=0, column=0, pady=(0, 10))
        image_container.grid_propagate(False)
        
        # Image display area with placeholder text
        self.image_label = ttk.Label(image_container, text="Click Browse to select an image")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Browse button - minimal style
        browse_button = ttk.Button(image_frame, text="Browse Image", 
                                 command=self.browse_image, style='Browse.TButton', width=20)
        browse_button.grid(row=1, column=0, pady=10)
        
        # Results section - minimal design
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar - clean look
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(results_frame, length=350, mode='determinate', 
                                      variable=self.progress_var)
        self.progress.grid(row=0, column=0, pady=(0, 15))
        
        # Prediction results - no frame, just labels
        result_container = ttk.Frame(results_frame)
        result_container.grid(row=1, column=0, pady=(0, 15))
        
        # Center-aligned prediction labels
        self.prediction_label = ttk.Label(result_container, text="", style='Result.TLabel')
        self.prediction_label.grid(row=0, column=0, pady=3)
        
        self.confidence_label = ttk.Label(result_container, text="", style='Info.TLabel')
        self.confidence_label.grid(row=1, column=0, pady=3)
        
        # Simplified table frame
        table_container = ttk.Frame(results_frame)
        table_container.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Clean, minimal treeview
        self.tree = ttk.Treeview(table_container, columns=('Denomination', 'Confidence'), 
                                height=3, style="Treeview", show='headings')
        self.tree.heading('Denomination', text='Denomination')
        self.tree.heading('Confidence', text='Confidence')
        self.tree.column('Denomination', anchor=tk.CENTER, width=175)
        self.tree.column('Confidence', anchor=tk.CENTER, width=175)
        self.tree.grid(row=0, column=0, pady=5)
        
    def browse_image(self):
        """Open file dialog to choose an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            self.load_and_predict(file_path)
    
    def load_and_predict(self, image_path):
        """Load image and make prediction"""
        # Load and display image
        image = Image.open(image_path)
        # Resize image for display while maintaining aspect ratio
        display_size = (300, 300)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        # Start prediction in a separate thread
        thread = threading.Thread(target=self.predict_image, args=(image_path,))
        thread.daemon = True
        thread.start()
    
    def analyze_image_features(self, img_array):
        """Analyze image features to detect if it matches Indian currency characteristics"""
        # Convert to float32 for calculations
        img = img_array.astype(np.float32)
        
        # Calculate color statistics
        mean_color = np.mean(img, axis=(0, 1))
        color_std = np.std(img, axis=(0, 1))
        
        # Indian currency notes have specific color patterns and variations
        # These thresholds are based on typical Indian currency characteristics
        color_variation = np.mean(color_std)
        is_color_valid = 0.15 <= color_variation <= 0.45
        
        # Check if the image has enough detail/texture (Indian notes have distinct patterns)
        detail_level = np.mean(np.abs(np.diff(img, axis=0))) + np.mean(np.abs(np.diff(img, axis=1)))
        has_enough_detail = detail_level > 0.1
        
        return is_color_valid and has_enough_detail

    def preprocess_image(self, img_array):
        """Enhanced preprocessing for Indian currency notes"""
        # Convert to float32 for better precision
        img = img_array.astype(np.float32)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Apply adaptive histogram equalization for better contrast
        for channel in range(3):
            img_channel = img[:, :, channel]
            p2, p98 = np.percentile(img_channel, (2, 98))
            img[:, :, channel] = np.clip(img_channel, p2, p98)
            img[:, :, channel] = (img_channel - p2) / (p98 - p2 + 1e-7)
        
        # Ensure values are in [0, 1]
        img = np.clip(img, 0, 1)
        
        return img

    def predict_image(self, image_path):
        """Make prediction on the image"""
        try:
            # Update progress bar
            self.progress_var.set(0)
            self.root.update_idletasks()
            
            # Load and preprocess image
            img = load_img(image_path, target_size=(self.image_size, self.image_size))
            img_array = img_to_array(img)
            
            # Apply enhanced preprocessing
            processed_img = self.preprocess_image(img_array)
            img_array = np.expand_dims(processed_img, axis=0)
            
            self.progress_var.set(50)
            self.root.update_idletasks()
            
            # Get prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Get top 2 predictions and their difference
            top_2_indices = np.argsort(predictions[0])[-2:]
            top_2_confidence = predictions[0][top_2_indices]
            confidence_diff = top_2_confidence[1] - top_2_confidence[0]
            
            self.progress_var.set(100)
            self.root.update_idletasks()
            
            # Update GUI with results
            self.root.after(0, self.update_results, predicted_class, confidence, predictions[0])
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def update_results(self, predicted_class, confidence, predictions):
        """Update GUI with prediction results"""
        # Clear previous entries in the tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Get sorted predictions and indices
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_predictions = predictions[sorted_indices]

        # Get denominations for top predictions
        top_denom = int(self.class_names[sorted_indices[0]])
        second_denom = int(self.class_names[sorted_indices[1]])

        # Define denomination groups and thresholds
        LOW_DENOMS = set()  # No low denominations anymore
        MID_DENOMS = {10, 20, 50}
        HIGH_DENOMS = {100, 200, 500}
        
        # Confidence thresholds for different denomination groups
        HIGH_VALUE_THRESHOLD = 0.40
        MID_VALUE_THRESHOLD = 0.35
        LOW_VALUE_THRESHOLD = 0.30
        
        # Determine which group the prediction falls into
        is_low_value_pred = top_denom in LOW_DENOMS
        is_mid_value_pred = top_denom in MID_DENOMS
        is_high_value_pred = top_denom in HIGH_DENOMS
        
        # Special handling for low-value notes (1, 2, 5)
        if is_low_value_pred:
            # Check if top two predictions are both low-value notes
            both_low_value = second_denom in LOW_DENOMS
            
            if both_low_value:
                # If comparing between 1, 2, 5 rupee notes
                confidence_threshold = LOW_VALUE_THRESHOLD
                diff_threshold = 0.10  # Need only 10% difference
                
                # Additional check for 5 rupee notes
                if top_denom == 5 and sorted_predictions[0] > 0.45:
                    diff_threshold = 0.08  # Even more lenient for ₹5 with good confidence
            else:
                confidence_threshold = LOW_VALUE_THRESHOLD
                diff_threshold = 0.15
        
        # Handle mid-value notes (10, 20, 50)
        elif is_mid_value_pred:
            confidence_threshold = MID_VALUE_THRESHOLD
            diff_threshold = 0.12
            
        # Handle high-value notes (100, 200, 500)
        else:
            both_high_value = second_denom in HIGH_DENOMS
            confidence_threshold = HIGH_VALUE_THRESHOLD
            
            if both_high_value:
                diff_threshold = 0.10
            else:
                diff_threshold = 0.15
                
            # Special case for 500 notes
            if top_denom == 500 and sorted_predictions[0] > 0.50:
                diff_threshold = 0.08
            
        # Calculate the difference between top predictions
        top_2_diff = sorted_predictions[0] - sorted_predictions[1]
        
        # Validation checks
        confidence_check = sorted_predictions[0] >= confidence_threshold
        difference_check = top_2_diff >= diff_threshold
        
        # Additional checks for confidence levels
        very_high_confidence = sorted_predictions[0] > 0.75
        good_confidence = sorted_predictions[0] > 0.45
        
        # Calculate the difference between top predictions
        top_2_diff = sorted_predictions[0] - sorted_predictions[1]
        
        # Decision making with special handling for ₹5 notes
        if very_high_confidence:
            # If we're very confident, accept the prediction
            is_valid = True
        elif good_confidence and top_denom == 5:
            # Special case for ₹5 notes with good confidence
            is_valid = top_2_diff >= 0.08  # Very lenient difference check for ₹5
        elif good_confidence and is_low_value_pred:
            # For low-value notes with good confidence
            is_valid = top_2_diff >= diff_threshold
        else:
            # Standard validation
            is_valid = sorted_predictions[0] >= confidence_threshold and top_2_diff >= diff_threshold
        
        if not is_valid:
            # Invalid prediction
            self.prediction_label.configure(text="Uncertain Prediction")
            detail_text = "Unable to make a confident prediction.\n"
            if not confidence_check:
                detail_text += f"Confidence too low: {sorted_predictions[0]:.2%}\n"
            if not difference_check:
                detail_text += f"Distinction too low: {top_2_diff:.2%}\n"
            detail_text += "\nPlease ensure you're using a clear image of an Indian currency note."
            self.confidence_label.configure(text=detail_text)
        else:
            # Valid prediction
            denomination = self.class_names[predicted_class]
            self.prediction_label.configure(text=f"Predicted Denomination: ₹{denomination}")
            # Display confidence as percentage
            confidence_pct = float(confidence) * 100
            self.confidence_label.configure(text=f"Confidence: {confidence_pct:.2f}%")
            
            # Speak the result
            self.speak_result(denomination)
            
        # Show top 3 predictions in the tree
        for i, idx in enumerate(sorted_indices[:3]):
            self.tree.insert('', i, values=(
                f"₹{self.class_names[idx]}", 
                f"{predictions[idx]:.2%}"
            ))
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        for i, idx in enumerate(top_3_indices):
            self.tree.insert('', i, values=(f"₹{self.class_names[idx]}", f"{predictions[idx]:.2%}"))
    
    def show_error(self, error_message):
        """Show error message"""
        self.prediction_label.configure(text="Error!")
        self.confidence_label.configure(text=error_message)
        
    def show_invalid_currency(self):
        """Show message for non-Indian currency"""
        self.prediction_label.configure(text="Not an Indian Currency Note!")
        self.confidence_label.configure(text="Please use only Indian currency notes for classification")
        # Clear previous entries in the tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.progress_var.set(100)
        
    def speak_result(self, denomination):
        """Speak the denomination in selected language"""
        if self.language.get() == "English":
            text = f"I've identified this as a ₹{denomination} note. For better results, please capture the picture in good lighting."
            lang = 'en'
        else:
            hindi_number = self.hindi_numbers[denomination]
            text = f"मैंने इसे {hindi_number}₹ का नोट पहचाना है। बेहतर परिणामों के लिए, कृपया अच्छी रोशनी में तस्वीर लें।"
            lang = 'hi'
            
        try:
            # Create and save audio with higher quality
            tts = gTTS(text=text, lang=lang)
            
            # Save and play the audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='w+b')
            temp_filename = temp_file.name
            temp_file.close()
            
            # Save audio to temporary file
            tts.save(temp_filename)
            
            def cleanup_audio():
                try:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except:
                    pass
                try:
                    os.unlink(temp_filename)
                except:
                    pass

            def play_audio():
                try:
                    # Reinitialize mixer if needed
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    
                    # Stop any playing audio first
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    
                    # Load and play at higher volume
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.set_volume(1.0)
                    pygame.mixer.music.play()
                    
                    # Schedule cleanup after playing (longer delay for longer message)
                    self.root.after(8000, cleanup_audio)
                except Exception as e:
                    print(f"Playback error: {e}")
                    cleanup_audio()
            
            # Add delay to ensure file is fully written and ready
            self.root.after(200, play_audio)
        except Exception as e:
            print(f"Audio error: {e}")

def main():
    root = tk.Tk()
    app = CurrencyClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()