# 🍲 Naija Food Classifier

An image classifier that recognises 10 popular Nigerian dishes from a photo. Upload or snap a picture and the app names the dish, with its top 3 guesses and a confidence score.

**🔗 Live demo:** <[Link](https://nigerian-food-classifier.streamlit.app/)>

![App screenshot](screenshot.png)

## Dishes it recognises

Akara · Amala · Efo Riro · Egusi Soup · Fried Rice · Jollof Rice · Moi Moi · Pounded Yam · Puff Puff · Suya

## Results

| Model | Input size | Test accuracy |
|---|---|---|
| EfficientNet-B0 (baseline) | 224 px | 83.8% |
| **ConvNeXt-Tiny (final)** | **288 px** | **93.2%** |

Evaluated on a held-out test set of 74 images (about 7 to 8 per class). Because the test set is small, treat the accuracy as an estimate rather than an exact figure.

The biggest improvement came from higher resolution and a stronger backbone. The baseline often confused akara with moi_moi (both are bean-based), and akara recall rose from 0.57 to 0.86 after the upgrade.

## How it was built

1. **Data cleaning:** scanned the dataset for corrupt files, removed exact duplicates (by file hash), and dropped images that appeared under more than one class.
2. **Split:** stratified 70% train / 15% validation / 15% test.
3. **Model:** transfer learning with a pretrained ConvNeXt-Tiny (PyTorch / torchvision), trained in two phases: first the new classification head, then fine-tuning the upper layers at a low learning rate.
4. **Augmentation:** random resized crop, flips, TrivialAugmentWide, random erasing; label smoothing in the loss.
5. **App:** a Streamlit interface that loads the model weights from Hugging Face.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, Streamlit, Hugging Face Hub

## Run it locally

```bash
git clone https://github.com/<[YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME](https://github.com/Sultanast01/nigerian-food-classifier.git)>.git
cd <YOUR_REPO_NAME>
pip install -r requirements.txt
streamlit run app.py
```

The model weights are downloaded automatically from Hugging Face the first time the app runs: [techwithst/nigerian-food-classifier](https://huggingface.co/techwithst/nigerian-food-classifier).

## Repository contents

```
app.py             Streamlit app
requirements.txt   Python dependencies
notebook.ipynb     Training notebook (data cleaning, training, evaluation)
README.md          This file
```

## Limitations

- The model only knows the 10 dishes above. A photo of any other food will get a wrong or low-confidence answer; the app warns when confidence is under 50%.
- The dataset is small (roughly 50 training images per class), so unusual angles, lighting, or plating can cause mistakes.
- The most common confusions are between similar-looking dishes, such as akara and moi_moi, or jollof rice and fried rice.

## Dataset

Trained on the Nigerian Food Dataset from Kaggle (by foyinsolaodekunle).

## Possible next steps

- Collect more photos per class, especially akara, moi_moi, and jollof rice
- Add more Nigerian dishes
- Collect user feedback on wrong predictions to retrain the model
