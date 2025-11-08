import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import sys


st.set_page_config(
    page_title="Seatbelt Detection",
    page_icon="🚗",
    layout="wide"
)

# --- Model Loading with Ultralytics YOLOv11 ---
@st.cache_resource  # Cache the model to load only once
def load_model():
    try:

        model = YOLO('yolov11.pt')
        return model, None
    except Exception as e:
        return None, str(e)

with st.spinner("🔄 Loading YOLOv11 model... (This may take a minute on first load)"):
    model, error = load_model()

if error:
    st.error(f"❌ Error loading the YOLOv11 model: {error}")
    st.warning("⚠️ Please ensure 'yolov11.pt' is in the correct directory and accessible.")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# --- Prediction Function ---
@st.cache_data
def predict(_model, _image):
    """
    Perform prediction on the image
    """
    try:
        # Convert PIL image to numpy array for Ultralytics
        if isinstance(_image, Image.Image):
            np_img = np.array(_image)
        else:
            np_img = _image
        
        # Run detection
        results = _model.predict(np_img)
        
        # Get result with bounding boxes as np array
        annotated_img = results[0].plot()
        return Image.fromarray(annotated_img), None
    except Exception as e:
        return None, str(e)

# --- Get Example Images ---
@st.cache_data
def get_example_images():
    """Load example images from the images folder"""
    examples = []
    image_folder = "images"
    
    if not os.path.exists(image_folder):
        return []
    
    try:
        for filename in sorted(os.listdir(image_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                examples.append(os.path.join(image_folder, filename))
        return examples
    except Exception as e:
        st.error(f"Error loading example images: {e}")
        return []

# --- Title and Description ---
st.title("🚗 Seatbelt Detection with YOLOv11")
st.markdown("""
Upload an image to detect seatbelt usage. The model will identify people wearing or not wearing seatbelts.
""")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_image = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing people in vehicles"
    )

# Process uploaded image
if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        with col1:
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        
        # Run prediction
        with st.spinner("🔍 Detecting seatbelts..."):
            result_image, pred_error = predict(model, image)
        
        if pred_error:
            st.error(f"❌ Error during prediction: {pred_error}")
        elif result_image is not None:
            with col2:
                st.subheader("🎯 Detection Results")
                st.image(result_image, caption="✅ Detected Image", use_container_width=True)
                
                import io
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="⬇️ Download Result",
                    data=byte_im,
                    file_name="seatbelt_detection_result.png",
                    mime="image/png"
                )
    except Exception as e:
        st.error(f"❌ Error processing uploaded image: {e}")
        st.warning("Please ensure the uploaded file is a valid image.")

# --- Example Images Section ---
st.markdown("---")
st.subheader("📚 Example Images")

show_examples = st.checkbox('Show example images from dataset')

if show_examples:
    example_images = get_example_images()
    
    if example_images:
        # Display examples in a grid
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, example_path in enumerate(example_images):
            try:
                img = Image.open(example_path)
                col_idx = idx % num_cols
                
                with cols[col_idx]:
                    st.image(img, caption=os.path.basename(example_path), use_container_width=True)
                    if st.button(f"Use this image", key=f"btn_{idx}"):
                        st.info("Please upload this image using the file uploader above.")
            except Exception as e:
                st.error(f"Could not load {os.path.basename(example_path)}: {e}")
    else:
        st.info("ℹ️ No example images found in the 'images' folder.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by YOLOv11 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)