import streamlit as st
import os
import numpy as np
from PIL import Image
from io import BytesIO
import joblib
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


# Page Config
st.set_page_config(page_title="Image Classifier", layout="wide")
st.title("🐼🧸 Custom Image Classifier")
st.write("Upload class images → Train models → Predict on new image with metrics.")


# Folders
folders = ['data/upload','models']
for f in folders:
    os.makedirs(f, exist_ok=True)


# Step 1: Upload class images with Webcam

st.header("Step 1: Upload Images for a Class")
class_name = st.text_input("Enter Class Name (e.g., pandas, bear)")

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload images for this class", 
    type=["jpg","jpeg","png"], 
    accept_multiple_files=True
)

# --- Webcam Capture ---
st.write("Or take pictures from your webcam")
cam_image = st.camera_input("Take a photo")

# --- Save Images Button ---
if st.button("Save Images"):
    if not class_name:
        st.error("Enter a class name")
    elif not uploaded_files and cam_image is None:
        st.error("Upload at least 1 image or capture from webcam")
    else:
        class_path = os.path.join("data/upload", class_name)
        os.makedirs(class_path, exist_ok=True)
        saved_count = 0

        # Save uploaded files
        if uploaded_files:
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                img.save(os.path.join(class_path, file.name))
                saved_count += 1

        # Save webcam image
        if cam_image is not None:
            img = Image.open(cam_image).convert("RGB")
            save_path = os.path.join(class_path, f"webcam_{len(os.listdir(class_path))+1}.jpg")
            img.save(save_path)
            saved_count += 1

        st.success(f"Saved {saved_count} images in '{class_name}'")


# Original Helper Functions

def load_images_labels(base_dir='data/upload'):
    X=[]
    y=[]
    classes = sorted(os.listdir(base_dir))
    for clss in classes:
        clss_path = os.path.join(base_dir,clss)
        if not os.path.isdir(clss_path):
            continue
        for fname in os.listdir(clss_path):
            fpath = os.path.join(clss_path,fname)
            try:
                img = Image.open(fpath).convert('RGB').resize((224,224))
                arr = np.array(img)
                X.append(arr)
                y.append(clss)
            except:
                pass
    X = np.array(X)
    y = np.array(y)
    return X, y

def flatten_X(X):
    return X.reshape(X.shape[0], -1)

def encode_y(y):
    le = LabelEncoder()
    y_encoder = le.fit_transform(y)
    y_categorical = to_categorical(y_encoder)
    return le, y_encoder, y_categorical

def train_cnn(X,y_categorical,epochs=5,save_path="models/cnn.keras", batch_size=16):
    input_shape = X.shape[1:]
    num_classes = y_categorical.shape[1]

    data_aug = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1)
    ])

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        layers.Input(shape=input_shape),
        data_aug,
        #layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(X, y_categorical, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        callbacks=[checkpoint, early])
    return model, history


# Evaluation Function

def eval_sklearn(model, X, y, le):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=le.classes_)
    return acc, cm, report

def eval_cnn(model, X, y_encoder, le):
    X = X.astype('float32')/255
    preds = model.predict(X)
    preds_idx = np.argmax(preds, axis=1)
    return eval_sklearn_simple(preds_idx, y_encoder, le)

def eval_sklearn_simple(preds_idx, y_encoder, le):
    acc = accuracy_score(y_encoder, preds_idx)
    cm = confusion_matrix(y_encoder, preds_idx)
    report = classification_report(y_encoder, preds_idx, target_names=le.classes_)
    return acc, cm, report


# Step 2: Train Models

st.header("Step 2: Train Models on Uploaded Data")
if st.button("Train Models"):
    X, y = load_images_labels()
    if len(X) < 1:
        st.error("No images found. Upload class images first!")
    else:
        le, y_encoder, y_categorical = encode_y(y)
        X_flat = flatten_X(X)
        X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoder, test_size=0.2, random_state=123)

        # Logistic
        st.info("Training Logistic Regression...")
        logistic_model = LogisticRegression(max_iter=500)
        logistic_model.fit(X_train, y_train)
        joblib.dump(logistic_model,"models/logistic.pkl")
        st.success("Logistic Regression saved")

        # Random Forest
        st.info("Training Random Forest...")
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model,"models/randomforest.pkl")
        st.success("Random Forest saved")

        # CNN
        st.info("Training CNN...")
        cnn_model, history = train_cnn(X.astype('float32'), y_categorical, epochs=5, batch_size=16)
        cnn_model.save("models/cnn.keras")
        st.success("CNN saved")

        # Save Label Encoder
        with open("models/label_encoder.pkl","wb") as f:
            pickle.dump(le,f)
        st.success("Label Encoder saved")

        
        # Evaluation Metrics
      
        st.subheader("✅ Training Metrics")
        st.write("**Logistic Regression Metrics**")
        acc, cm, report = eval_sklearn(logistic_model, X_train, y_train, le)
        st.write(f"Accuracy: {acc}")
        st.write("Confusion Matrix:")
        st.write(cm)
        st.text(report)

        st.write("**Random Forest Metrics**")
        acc, cm, report = eval_sklearn(rf_model, X_train, y_train, le)
        st.write(f"Accuracy: {acc}")
        st.write("Confusion Matrix:")
        st.write(cm)
        st.text(report)

        st.write("**CNN Metrics**")
        acc, cm, report = eval_cnn(cnn_model, X, y_encoder, le)
        st.write(f"Accuracy: {acc}")
        st.write("Confusion Matrix:")
        st.write(cm)
        st.text(report)


# Step 3: Predict Single Image

st.header("Step 3: Predict on New Image")
uploaded_test = st.file_uploader("Upload Test Image", type=["jpg","jpeg","png"], key="predict")
if uploaded_test:
    img = Image.open(uploaded_test)
    st.image(img, caption="Uploaded Test Image", width=300)

    arr = np.array(img.resize((224,224)))
    arr_flat = arr.reshape(1,-1)
    arr_cnn = arr.reshape(1,224,224,3).astype('float32')/255.0

    # Load models
    log_model = joblib.load("models/logistic.pkl")
    rf_model = joblib.load("models/randomforest.pkl")
    cnn_model = load_model("models/cnn.keras")
    with open("models/label_encoder.pkl","rb") as f:
        le = pickle.load(f)

    # Predictions
    p_log = log_model.predict(arr_flat)[0]
    p_rf = rf_model.predict(arr_flat)[0]
    p_cnn_idx = np.argmax(cnn_model.predict(arr_cnn)[0])

    st.subheader("Predictions")
    st.write(f"**Logistic Regression:** {le.inverse_transform([p_log])[0]}")
    st.write(f"**Random Forest:** {le.inverse_transform([p_rf])[0]}")
    st.write(f"**CNN (ResNet50):** {le.inverse_transform([p_cnn_idx])[0]}")
