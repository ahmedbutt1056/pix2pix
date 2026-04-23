import io
import torch
import torch.nn as nn
import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Pix2Pix Image Translation",
    page_icon="🎨",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111827, #1f2937);
        color: white;
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 22px;
        padding: 1.2rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.30);
    }

    .info-box {
        background: rgba(96, 165, 250, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-top: 0.5rem;
        color: #dbeafe;
    }

    .footer-note {
        text-align: center;
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">🎨 Pix2Pix Image Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload an image and generate the translated output using your trained Pix2Pix generator from Hugging Face.</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Settings
# -----------------------------
img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model Classes
# Exact architecture based on your notebook app.py cell
# -----------------------------
class Down(nn.Module):
    def __init__(self, a, b, norm=True):
        super().__init__()
        arr = [nn.Conv2d(a, b, 4, 2, 1, bias=False)]
        if norm:
            arr.append(nn.BatchNorm2d(b))
        arr.append(nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(*arr)

    def forward(self, x):
        return self.main(x)


class Up(nn.Module):
    def __init__(self, a, b, drop=False):
        super().__init__()
        arr = [
            nn.ConvTranspose2d(a, b, 4, 2, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(True)
        ]
        if drop:
            arr.append(nn.Dropout(0.5))
        self.main = nn.Sequential(*arr)

    def forward(self, x):
        return self.main(x)


class Gen(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = Down(3, 64, norm=False)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)
        self.d5 = Down(512, 512)
        self.d6 = Down(512, 512)
        self.d7 = Down(512, 512)

        self.bot = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.ReLU(True)
        )

        self.u1 = Up(512, 512, drop=True)
        self.u2 = Up(1024, 512, drop=True)
        self.u3 = Up(1024, 512, drop=True)
        self.u4 = Up(1024, 512)
        self.u5 = Up(1024, 256)
        self.u6 = Up(512, 128)
        self.u7 = Up(256, 64)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        b = self.bot(d7)

        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))

        out = self.last(torch.cat([u7, d1], dim=1))
        return out

# -----------------------------
# Image Preprocess / Postprocess
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.tensor(arr).unsqueeze(0)
    return tensor


def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor * 0.5) + 0.5
    tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)

# -----------------------------
# Load Generator from Hugging Face
# -----------------------------
@st.cache_resource
def load_model():
    repo_id = "supremeproducts45/generatorpix2pix"

    possible_files = [
        
        
        "model.pth",
        
    ]

    model_path = None
    last_error = None

    for file_name in possible_files:
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=file_name)
            break
        except Exception as e:
            last_error = e

    if model_path is None:
        raise FileNotFoundError(
            "Generator weight file not found in Hugging Face repo. "
            "Make sure your repo contains g_final.pth or update the filename list."
        ) from last_error

    model = Gen().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model Details")
    st.write("**Generator Repo**")
    st.code("supremeproducts45/generatorpix2pix")

    st.write("**Discriminator Repo**")
    st.code("supremeproducts45/discriminatorpix2pix")

    st.markdown(
        """
        <div class="info-box">
        For inference, only the generator is used.<br><br>
        The discriminator is usually only needed during training.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.write("**Image Size**")
    st.write("128 × 128")

# -----------------------------
# Main Layout
# -----------------------------
left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Input Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🚀 Generate Output")
    generate_btn = st.button("Generate Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Inference
# -----------------------------
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(input_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if generate_btn:
        try:
            with st.spinner("Loading model and generating output..."):
                model = load_model()
                x = preprocess_image(input_image).to(device)

                with torch.no_grad():
                    y = model(x)

                output_image = postprocess_tensor(y)

            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Generated Output")
                st.image(output_image, use_container_width=True)

                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                st.download_button(
                    label="⬇ Download Output",
                    data=buf.getvalue(),
                    file_name="pix2pix_output.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            st.success("Image generated successfully.")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Upload an image first, then click Generate Image.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer-note">Built with Streamlit • Powered by Pix2Pix Generator on Hugging Face</div>',
    unsafe_allow_html=True
)