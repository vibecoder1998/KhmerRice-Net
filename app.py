"""
Main Streamlit application for Rice Disease Classification
Enhanced UI/UX version
"""
from pathlib import Path
import streamlit as st
from PIL import Image
import time

from src.inference import load_model_and_metadata, predict
from utils.label_map import DISEASE_INFO


# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #28a745;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .advice-item {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .source-link {
        color: #0066cc;
        text-decoration: none;
    }
    .source-link:hover {
        text-decoration: underline;
    }
    .stProgress > div > div > div {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)


def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_confidence_emoji(confidence):
    """Get emoji based on confidence level"""
    if confidence >= 0.8:
        return "✅"
    elif confidence >= 0.5:
        return "⚠️"
    else:
        return "❌"


# MAIN APP
def main():
    st.set_page_config(
        page_title="Rice Disease Classifier | កម្មវិធីវិនិច្ឆ័យជំងឺស្លឹកស្រូវ",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for settings and info
    with st.sidebar:
        st.markdown("### 🌾 Settings")
        
        # Language selector
        lang = st.radio(
            "ជ្រើសរើសភាសា / Choose Language",
            ["ខ្មែរ", "English"],
            index=0
        )
        lang_key = "km" if lang == "ខ្មែរ" else "en"
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        about_text = (
            "កម្មវិធីនេះប្រើ AI ដើម្បីវិនិច្ឆ័យជំងឺស្លឹកស្រូវ។ "
            "បង្ហោះរូបភាពស្លឹកស្រូវដើម្បីចាប់ផ្តើម។"
            if lang_key == "km"
            else "This app uses AI to classify rice leaf diseases. "
            "Upload a rice leaf image to get started."
        )
        st.info(about_text)
        
        st.markdown("---")
        
        # Supported diseases
        st.markdown("### 📋 Supported Diseases")
        disease_list = [
            "Bacterial Leaf Blight",
            "Brown Spot",
            "Leaf Blast",
            "Leaf Scald",
            "Sheath Blight",
            "Healthy"
        ]
        for disease in disease_list:
            st.markdown(f"• {disease}")

    # Main content area
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Rice Leaf Disease Classifier</h1>
        <h2>កម្មវិធីវិនិច្ឆ័យជំងឺស្លឹកស្រូវ</h2>
    </div>
    """, unsafe_allow_html=True)

    # Description
    desc = (
        "បង្ហោះរូបភាពស្លឹកស្រូវរបស់អ្នក ហើយម៉ូឌែល AI នឹងវិភាគ និងវិនិច្ឆ័យជំងឺដែលអាចកើតឡើង។ "
        "យើងនឹងផ្តល់ឱ្យអ្នកនូវព័ត៌មានលម្អិត និងការណែនាំសម្រាប់ការគ្រប់គ្រង។"
        if lang_key == "km"
        else "Upload your rice leaf image and our AI model will analyze and classify any potential diseases. "
        "We'll provide you with detailed information and management recommendations."
    )
    st.markdown(f"<p style='font-size: 1.1em; text-align: center; color: #666;'>{desc}</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Load model (cached)
    @st.cache_resource
    def load_model():
        try:
            return load_model_and_metadata()
        except FileNotFoundError as e:
            st.error(str(e))
            return None, None, None

    model, classes, transform_info = load_model()
    
    if model is None:
        st.error("Model could not be loaded. Please check the model file.")
        st.stop()

    # Upload section
    st.markdown("### 📤 Upload Image")
    upload_label = (
        "បង្ហោះរូបភាពស្លឹកស្រូវ"
        if lang_key == "km"
        else "Upload a rice leaf image"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader(
            upload_label,
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        
        # Display image in a nice container
        st.markdown("### 📷 Uploaded Image")
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption="", use_container_width=True)
        
        st.markdown("---")

        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            btn_text = (
                "🔍 វាយតម្លៃជំងឺ"
                if lang_key == "km"
                else "🔍 Analyze Disease"
            )
            predict_btn = st.button(
                btn_text,
                type="primary",
                use_container_width=True
            )

        if predict_btn:
            # Prediction with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(
                "កំពុងវិភាគ..." if lang_key == "km" else "Analyzing image..."
            )
            progress_bar.progress(20)
            time.sleep(0.3)
            
            top_label, confidence, all_probs = predict(
                model, image, classes, top_k=6
            )
            
            progress_bar.progress(80)
            time.sleep(0.2)
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.markdown("---")

            # Get disease info
            if top_label in DISEASE_INFO:
                info = DISEASE_INFO[top_label][lang_key]
            else:
                info = {
                    "name": top_label.replace("_", " ").title(),
                    "advice": [
                        "Please consult with agricultural experts for proper diagnosis." if lang_key == "en"
                        else "សូមពិគ្រោះជាមួយអ្នកជំនាញកសិកម្មសម្រាប់ការវិនិច្ឆ័យត្រឹមត្រូវ។"
                    ],
                    "sources": {
                        "IRRI – Knowledge Bank": "https://www.knowledgebank.irri.org",
                        "MAFF Cambodia": "http://www.maff.gov.kh/" if lang_key == "km" else "https://gda.maff.gov.kh/",
                    }
                }
                st.warning(
                    f"⚠️ Unknown disease label: {top_label}" if lang_key == "en"
                    else f"⚠️ ស្លាកជំងឺមិនស្គាល់: {top_label}"
                )

            # Main prediction result card
            st.markdown("### 🎯 Prediction Results")
            
            # Prediction card with metrics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                result_header = (
                    f"**{info['name']}**"
                    if lang_key == "km"
                    else f"**{info['name']}**"
                )
                st.markdown(f"### {result_header}")
            
            with col2:
                confidence_emoji = get_confidence_emoji(confidence)
                confidence_class = get_confidence_color(confidence)
                confidence_display = f"{confidence*100:.1f}%"
                st.metric(
                    "Confidence" if lang_key == "en" else "ភាពជឿជាក់",
                    f"{confidence_emoji} {confidence_display}"
                )
                # Progress bar for confidence
                st.progress(confidence)

            st.markdown("---")

            # Detailed probabilities in expandable section
            with st.expander(
                "📊 View All Probabilities" if lang_key == "en" else "📊 មើលអត្រាទាំងអស់",
                expanded=False
            ):
                prob_header = (
                    "អត្រាក្នុងចំណាត់ថ្នាក់"
                    if lang_key == "km"
                    else "Class Probabilities"
                )
                st.markdown(f"**{prob_header}:**")
                
                for i, (label, prob) in enumerate(all_probs):
                    if label in DISEASE_INFO:
                        name = DISEASE_INFO[label][lang_key]["name"]
                    else:
                        name = label.replace("_", " ").title()
                    
                    # Color code based on probability
                    if i == 0:
                        bar_color = "#28a745"
                    elif prob > 0.1:
                        bar_color = "#ffc107"
                    else:
                        bar_color = "#6c757d"
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        st.write(f"{prob*100:.1f}%")
                    with col3:
                        st.progress(prob)

            st.markdown("---")

            # Management recommendations
            st.markdown("### 💡 Management Recommendations")
            advice_header = (
                "សកម្មភាពណែនាំសម្រាប់ស្រែ"
                if lang_key == "km"
                else "Recommended Field Actions"
            )
            st.markdown(f"**{advice_header}:**")
            
            for i, tip in enumerate(info["advice"], 1):
                st.markdown(f"""
                <div class="advice-item">
                    <strong>{i}.</strong> {tip}
                </div>
                """, unsafe_allow_html=True)

            # Information sources
            if "sources" in info and info["sources"]:
                st.markdown("---")
                st.markdown("### 📚 Information Sources")
                sources_header = (
                    "ប្រភពព័ត៌មាន"
                    if lang_key == "km"
                    else "Learn More"
                )
                st.markdown(f"**{sources_header}:**")
                
                source_cols = st.columns(len(info["sources"]))
                for idx, (source_name, source_url) in enumerate(info["sources"].items()):
                    with source_cols[idx]:
                        st.markdown(
                            f'<a href="{source_url}" target="_blank" class="source-link">🔗 {source_name}</a>',
                            unsafe_allow_html=True
                        )

            # Success message
            st.success(
                "✅ Analysis complete! Use the recommendations above to manage your rice field."
                if lang_key == "en"
                else "✅ ការវិភាគបានបញ្ចប់! ប្រើការណែនាំខាងលើដើម្បីគ្រប់គ្រងស្រែរបស់អ្នក។"
            )

    else:
        # Instructions when no image uploaded
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        instructions = [
            "1. Click 'Browse files' or drag and drop an image above",
            "2. Make sure the image shows a clear view of the rice leaf",
            "3. Click 'Analyze Disease' to get predictions",
            "4. Review the results and recommendations"
        ] if lang_key == "en" else [
            "1. ចុច 'Browse files' ឬទាញដាក់រូបភាពខាងលើ",
            "2. ត្រូវបានត្រូវរូបភាពបង្ហាញស្លឹកស្រូវឱ្យច្បាស់",
            "3. ចុច 'វាយតម្លៃជំងឺ' ដើម្បីទទួលការព្យាករណ៍",
            "4. ពិនិត្យលទ្ធផល និងការណែនាំ"
        ]
        
        for instruction in instructions:
            st.markdown(f"• {instruction}")
        
        # Example image placeholder
        st.markdown("---")
        st.info(
            "💡 **Tip:** For best results, use clear, well-lit images of rice leaves showing disease symptoms."
            if lang_key == "en"
            else "💡 **ព័ត៌មាន:** សម្រាប់លទ្ធផលល្អបំផុត ប្រើរូបភាពច្បាស់ និងមានពន្លឺល្អនៃស្លឹកស្រូវដែលបង្ហាញរោគសញ្ញា។"
        )


if __name__ == "__main__":
    main()
