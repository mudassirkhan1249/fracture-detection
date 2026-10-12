import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="Bone Fracture Detection System",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        color: #1976D2;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 25px;
    }
    [data-testid="stCameraInput"] video {
        width: 100% !important;
        height: auto !important;
        border-radius: 10px;
    }
    [data-testid="stImage"] img {
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load YOLO Model securely based on your folder structure
@st.cache_resource
def load_cached_resources():
  possible_model_paths = [
      os.path.join("..", "model", "model.pt"),
      os.path.join("model", "model.pt"),
      "model.pt",
  ]
  model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)
  if not model_path:
    raise FileNotFoundError(
        "Model file (model.pt) nahi mili. Apne folder structure ko check karein."
    )
  return YOLO(model_path)


try:
  model = load_cached_resources()
except Exception as e:
  st.error(f"Error loading model: {e}")
  st.stop()

# App Header
st.markdown(
    '<p class="main-title">🦴 Bone Fracture Detection System</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Upload an X-ray image or capture one using your'
    " device camera to instantly check for fractures.</p>",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
  st.image("https://img.icons8.com/color/96/skeleton-bone.png", width=80)
  st.markdown("### 📋 Usage Guidelines")
  st.markdown("""
    1. **Clear X-ray:** Ensure proper contrast and visibility.
    2. **Focus:** Keep the bone region centered in the frame.
    3. **Background:** Avoid reflections or blurry angles.
    4. **Input Options:** Use live camera or upload from gallery.
    """)
  st.markdown("---")
  st.markdown("### 🛠️ Tech Stack")
  st.markdown(
      "- **Model:** YOLO11-cls (PyTorch)\n- **Frontend:** Streamlit\n- **UI"
      " Fix:** Responsive CSS"
  )

# Main Content Layout
tab1, tab2 = st.tabs(["📸 Live Camera Capture", "📁 Gallery Upload"])
image = None

with tab1:
  st.markdown("### Device Camera")
  camera_image = st.camera_input("Capture a live photo of the X-ray")
  if camera_image is not None:
    image = Image.open(camera_image)

with tab2:
  st.markdown("### File Upload")
  uploaded_file = st.file_uploader(
      "Choose an X-ray image from your device...",
      type=["jpg", "jpeg", "png", "webp"],
  )
  if uploaded_file is not None:
    image = Image.open(uploaded_file)

if image is not None:
  st.markdown("---")
  col1, col2 = st.columns([1, 1], gap="large")

  with col1:
    st.markdown("### Selected X-ray Image")
    st.image(image, caption="Target X-ray Image", use_container_width=True)

  with col2:
    st.markdown("### 🔍 Model Analysis")
    with st.spinner("Analyzing the X-ray, please wait..."):
      results = model(image)
      probs = results[0].probs
      pred_probabilities = probs.data.cpu().numpy()
      top1_idx = probs.top1
      confidence = float(probs.top1conf.item()) * 100
      class_names = results[0].names
      predicted_class_name = class_names[top1_idx]

    st.success("Analysis complete!")
    st.markdown(f"#### 🧬 Predicted Result:")
    if "fractured" in predicted_class_name.lower():
      st.error(f"**{predicted_class_name}**")
    else:
      st.success(f"**{predicted_class_name}**")

    st.markdown(f"#### 📊 Confidence Score:")
    st.metric(label="Prediction Confidence", value=f"{confidence:.2f}%")

  st.markdown("---")
  st.markdown("### 📈 Class Probabilities Breakdown")

  sorted_indices = np.argsort(pred_probabilities)
  sorted_classes = [class_names[i] for i in sorted_indices]
  sorted_scores = [float(pred_probabilities[i]) * 100 for i in sorted_indices]

  # Transparent background styling so it adapts to Dark/Light themes automatically
  fig, ax = plt.subplots(figsize=(10, 2.8))
  fig.patch.set_alpha(0.0)
  ax.patch.set_alpha(0.0)

  bars = ax.barh(sorted_classes, sorted_scores, color="#339af0")
  ax.set_xlabel("Confidence (%)", fontsize=11, fontweight="bold", color="gray")
  ax.set_xlim(0, 100)
  ax.set_title(
      "Model Prediction Probabilities", fontsize=12, fontweight="bold", color="gray"
  )

  ax.tick_params(colors="gray", labelsize=10)
  for spine in ax.spines.values():
    spine.set_color("#495057")

  for bar in bars:
    width = bar.get_width()
    ax.annotate(
        f"{width:.2f}%",
        xy=(width, bar.get_y() + bar.get_height() / 2),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontweight="bold",
        color="gray",
    )

  plt.tight_layout()
  st.pyplot(fig, use_container_width=True)

else:
  st.info(
      "💡 To get started, please use the tabs above to either capture an"
      " X-ray using your **Live Camera** or **Upload** an image from your"
      " gallery."
  )