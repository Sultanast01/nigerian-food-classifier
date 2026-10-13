import torch
import torch.nn as nn
import streamlit as st
from PIL import Image, ImageOps
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

IMG_SIZE = 288
HF_REPO = "techwithst/nigerian-food-classifier"
HF_FILE = "nigerian_food_classifier_fp16.pt"   # keep whatever filename you already use

st.set_page_config(page_title="Naija Food Classifier", page_icon="🍲", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero {
    background: linear-gradient(135deg, #0f9d58, #f4b400);
    padding: 1.6rem 1rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; color: white; font-size: 2.1rem;}
.hero p {margin: .4rem 0 0; color: white; opacity: .95;}
.result {
    border-radius: 16px;
    padding: 1rem 1.2rem;
    background: rgba(15, 157, 88, .12);
    border: 1px solid rgba(15, 157, 88, .45);
}
.result .label {font-size: .75rem; text-transform: uppercase; letter-spacing: .09em; opacity: .7;}
.result .dish {font-size: 1.9rem; font-weight: 700; line-height: 1.2;}
.result .conf {opacity: .8; margin-top: .2rem;}
</style>
""", unsafe_allow_html=True)

DESCRIPTIONS = {
    "akara": "Deep-fried bean cakes, crispy outside and soft inside.",
    "amala": "A smooth, dark swallow made from yam or cassava flour.",
    "efo_riro": "A rich Yoruba vegetable soup, usually with peppers and assorted meat or fish.",
    "egusi_soup": "A hearty soup thickened with ground melon seeds.",
    "fried_rice": "Rice stir-fried with vegetables, often served with chicken or plantain.",
    "jollof_rice": "Party-favourite rice cooked in a spicy tomato and pepper base.",
    "moi_moi": "Steamed bean pudding, soft and smooth, often wrapped in leaves or foil.",
    "pounded_yam": "A stretchy swallow made by pounding boiled yam.",
    "puff_puff": "Soft, sweet, deep-fried dough balls.",
    "suya": "Spicy grilled skewered meat dusted with yaji pepper spice.",
}

eval_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner="Loading the model (first run takes a moment)...")
def load_model():
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    ckpt = torch.load(path, map_location="cpu")
    class_names = ckpt["class_names"]

    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(0.3),
        nn.Linear(in_features, len(class_names)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.float().eval()
    return model, class_names


def predict(model, class_names, img):
    with torch.no_grad():
        probs = torch.softmax(model(eval_tf(img).unsqueeze(0)), dim=1)[0]
    top = probs.topk(3)
    return [class_names[i] for i in top.indices], [p.item() for p in top.values]


def pretty(key):
    return key.replace("_", " ").title()


model, class_names = load_model()

# ---------- header ----------
st.markdown(
    '<div class="hero"><h1>🍲 Naija Food Classifier</h1>'
    "<p>Snap or upload a photo and find out which Nigerian dish it is.</p></div>",
    unsafe_allow_html=True,
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("An image classifier trained on Nigerian dishes, built with PyTorch (ConvNeXt-Tiny).")
    st.subheader("Dishes it knows")
    for c in class_names:
        st.write("• " + pretty(c))
    st.caption("Photos of other foods may get a wrong or low-confidence answer.")

# ---------- input ----------
tab_upload, tab_camera = st.tabs(["📁 Upload a photo", "📷 Take a photo"])
source = None

with tab_upload:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    if uploaded:
        source = uploaded

with tab_camera:
    if st.toggle("Turn on camera"):
        shot = st.camera_input("Take a picture", label_visibility="collapsed")
        if shot:
            source = shot

# ---------- result ----------
if source is not None:
    img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")   # fixes rotated phone photos

    with st.spinner("Identifying the dish..."):
        names, probs = predict(model, class_names, img)

    col_img, col_res = st.columns(2, gap="medium")

    with col_img:
        st.image(img, use_container_width=True)

    with col_res:
        if probs[0] < 0.5:
            st.warning(f"Not very sure. Best guess: **{pretty(names[0])}** ({probs[0]:.0%}). "
                       "This may be a dish the model doesn't know.")
        else:
            st.markdown(
                f'<div class="result"><div class="label">Looks like</div>'
                f'<div class="dish">{pretty(names[0])}</div>'
                f'<div class="conf">{probs[0]:.0%} confident</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(DESCRIPTIONS.get(names[0], ""))

    st.subheader("Top 3 guesses")
    for n, p in zip(names, probs):
        st.progress(float(p), text=f"{pretty(n)}  ·  {p:.0%}")
