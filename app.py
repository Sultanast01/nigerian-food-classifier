import torch
import torch.nn as nn
import streamlit as st
from PIL import Image, ImageOps
from torchvision import models, transforms

IMG_SIZE = 288
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="techwithst/nigerian-food-classifier",
    filename="nigerian_food_classifier_fp16.pt",   # exactly as named on Hugging Face
)   # or the fp16 version if you made one

st.set_page_config(page_title="Nigerian Food Classifier", page_icon="🍲")


@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
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


eval_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model, class_names = load_model()

st.title("🍲 Nigerian Food Classifier")

st.write(
    "Upload a photo of a Nigerian dish and the model will identify it. "
    "It can recognize dishes such as Akara, Amala, Efo Riro, Egusi Soup, "
    "Fried Rice, Jollof Rice, Moi Moi, Pounded Yam, Puff Puff, and Suya."
)


st.caption("It knows these dishes: " + ", ".join(c.replace("_", " ") for c in class_names))

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")   # fixes rotated phone photos
    st.image(img, use_container_width=True)

    with torch.no_grad():
        x = eval_tf(img).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1)[0]

    top = probs.topk(3)
    best_name = class_names[top.indices[0]].replace("_", " ")
    best_prob = top.values[0].item()

    if best_prob < 0.5:
        st.warning(f"Not very sure. Best guess: **{best_name}** ({best_prob:.0%}). "
                   "The photo may show a dish the model doesn't know.")
    else:
        st.success(f"**{best_name}** ({best_prob:.0%} confident)")

    st.subheader("Top 3 guesses")
    for p, i in zip(top.values, top.indices):
        st.write(class_names[i].replace("_", " ").title())
        st.progress(float(p))
