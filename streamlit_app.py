import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
from io import BytesIO
import tempfile
import os

# Page configuration
st.set_page_config(page_title="Currency Classifier", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = load_model('best_currency_classifier.h5')
    st.session_state.image_size = 224
    st.session_state.class_names = ['10', '20', '50', '100', '200', '500']
    st.session_state.hindi_numbers = {
        '10': 'दस',
        '20': 'बीस',
        '50': 'पचास',
        '100': 'सौ',
        '200': 'दो सौ',
        '500': 'पाँच सौ'
    }

def preprocess_image(img_array):
    """Enhanced preprocessing for Indian currency notes - SAME AS TKINTER GUI"""
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

def predict_currency(image_path):
    """Make prediction on the image"""
    try:
        # Handle both file paths and file objects
        if isinstance(image_path, str):
            img = load_img(image_path, target_size=(st.session_state.image_size, st.session_state.image_size))
            img_array = img_to_array(img)
        else:
            # It's a file-like object from Streamlit - use Keras load_img for consistency
            import tempfile
            import os
            
            # Get file extension from the uploaded file
            file_name = image_path.name if hasattr(image_path, 'name') else 'image.jpg'
            file_ext = os.path.splitext(file_name)[1] if file_name else '.jpg'
            
            # Save uploaded file to temporary location with correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(image_path.getbuffer())
                tmp_path = tmp.name
            
            # Load using Keras (SAME AS TKINTER)
            img = load_img(tmp_path, target_size=(st.session_state.image_size, st.session_state.image_size))
            img_array = img_to_array(img)
            
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        # Apply enhanced preprocessing (SAME AS TKINTER)
        processed_img = preprocess_image(img_array)
        img_array = np.expand_dims(processed_img, axis=0)
        
        # Get predictions - model already outputs probabilities!
        predictions_raw = st.session_state.model.predict(img_array, verbose=0)[0]
        
        predicted_class = np.argmax(predictions_raw)
        
        # Get sorted predictions with proper indexing
        sorted_indices = np.argsort(predictions_raw)[::-1]
        
        # Top 2 difference
        top_2_diff = predictions_raw[sorted_indices[0]] - predictions_raw[sorted_indices[1]]
        
        # Simple validation - just check if top prediction is clear winner
        is_valid = top_2_diff >= 0.01  # Just need 1% difference
        
        # Return raw predictions (not sorted) so we can access by index correctly
        return is_valid, predicted_class, predictions_raw, sorted_indices
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return False, None, None, None

def speak_result(denomination, language):
    """Generate audio using gTTS"""
    try:
        if language == "English":
            text = f"I've identified this as a {denomination}₹ note. For better results, please capture the picture in good lighting."
            lang = 'en'
        else:
            hindi_number = st.session_state.hindi_numbers[str(denomination)]
            text = f"मैंने इसे {hindi_number}₹ का नोट पहचाना है। बेहतर परिणामों के लिए, कृपया अच्छी रोशनी में तस्वीर लें।"
            lang = 'hi'
        
        try:
            # Create audio using gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to temp file
            import tempfile
            import os
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            tts.save(temp_path)
            
            # Read and play the audio (keep file temporarily)
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Play audio in background using HTML audio
            audio_base64 = __import__('base64').b64encode(audio_bytes).decode()
            audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg"></audio>'
            st.html(audio_html)
            
            # Cleanup
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            pass
        
    except Exception as e:
        pass

# Main app
st.title("🪙 Currency Classifier")
st.markdown("Classify Indian currency notes using AI")

# Language selection
col1, col2 = st.columns(2)
with col1:
    language = st.radio("Language / भाषा:", ["English", "हिंदी"], horizontal=True)

# Image upload
st.markdown("---")
uploaded_file = st.file_uploader("Upload a currency note image", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    if st.button("🔍 Classify Currency", use_container_width=True):
        with st.spinner("Analyzing image..."):
            is_valid, predicted_class, predictions, sorted_indices = predict_currency(uploaded_file)
        
        if is_valid and predicted_class is not None:
            denomination = st.session_state.class_names[predicted_class]
            
            # Display result
            st.markdown("---")
            st.markdown("### ✅ Classification Result")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Denomination", f"₹{denomination}")
            with col2:
                # Get confidence from raw predictions array (SAME AS TKINTER)
                confidence_value = float(predictions[predicted_class])
                st.metric("Confidence", f"{confidence_value*100:.2f}%")
            
            # Show top 3 predictions
            st.markdown("### Top Predictions")
            top_3_indices = sorted_indices[:3]
            
            prediction_data = []
            for idx in top_3_indices:
                conf_value = float(predictions[idx])
                prediction_data.append({
                    "Denomination": f"₹{st.session_state.class_names[idx]}",
                    "Confidence": f"{conf_value*100:.2f}%"
                })
            
            st.table(prediction_data)
            
            # Play audio automatically (silently)
            speak_result(denomination, language)
            
        else:
            st.markdown("---")
            st.error("❌ Unable to make a confident prediction. Please ensure you're using a clear image of an Indian currency note.")
else:
    st.info("👆 Upload an image of an Indian currency note to get started")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><small>Powered by TensorFlow • Built with Streamlit</small></div>", unsafe_allow_html=True)
