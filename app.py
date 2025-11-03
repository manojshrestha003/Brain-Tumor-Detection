from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Create FastAPI app
app = FastAPI()

# Allow all origins for now
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("model.h5")

# Define input size to match training
IMG_SIZE = (150, 150)

# Class labels same as training order
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Descriptions and recommendations for each class
class_info = {
    "glioma": {
        "description": "Glioma tumors arise from glial cells in the brain.",
        "recommendation": "Consult a neuro-oncologist for diagnosis and treatment options."
    },
    "meningioma": {
        "description": "Meningioma tumors originate from the meninges, the protective layers of the brain.",
        "recommendation": "Seek medical evaluation; surgical removal may be required depending on size and symptoms."
    },
    "notumor": {
        "description": "No visible signs of brain tumor detected.",
        "recommendation": "If symptoms persist, consult a neurologist for further evaluation."
    },
    "pituitary": {
        "description": "Pituitary tumors develop in the pituitary gland at the base of the brain.",
        "recommendation": "Consult an endocrinologist or neurosurgeon for assessment and possible treatment."
    }
}

@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Correct shape for model

        # Prediction
        prediction = model.predict(img_array)
        predicted_class_idx = int(np.argmax(prediction))
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(np.max(prediction))

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "description": class_info[predicted_class]["description"],
            "recommendation": class_info[predicted_class]["recommendation"]
        }

    except Exception as e:
        return {"error": str(e)}
