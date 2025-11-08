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
        "description": "Gliomas are tumors that develop from glial cells, which support and protect neurons in the brain. They can vary in aggressiveness and may affect brain function depending on their location and size.",
        "recommendation": "It is important to consult a neuro-oncologist for a detailed evaluation, which may include MRI scans and biopsy. Treatment options could include surgery, radiation therapy, or chemotherapy based on the tumor type and stage."
    },
    "meningioma": {
        "description": "Meningiomas are typically slow-growing tumors that arise from the meninges, the protective membranes surrounding the brain and spinal cord. While often benign, they can cause pressure on nearby brain structures.",
        "recommendation": "Seek medical assessment from a neurosurgeon. Depending on size, location, and symptoms, management may involve careful monitoring or surgical removal. Radiation therapy may be considered in some cases."
    },
    "notumor": {
        "description": "No signs of a brain tumor were detected in the scan. This indicates that the brain structure appears normal with no visible abnormal growths.",
        "recommendation": "If you continue to experience symptoms such as headaches, seizures, or neurological changes, it is advised to consult a neurologist for further evaluation and appropriate diagnostic tests."
    },
    "pituitary": {
        "description": "Pituitary tumors are abnormal growths in the pituitary gland, a small gland at the base of the brain that regulates hormones affecting growth, metabolism, and reproduction. Tumors may impact hormone balance and vision.",
        "recommendation": "Consult an endocrinologist or neurosurgeon for comprehensive evaluation. Treatment may include medication, surgical removal, or radiation therapy depending on tumor size, type, and hormone activity."
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

        img_array = np.array(image)
        img_array = img_array.reshape(1, 150, 150, 3) 
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
