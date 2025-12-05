import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
import io
from contextlib import redirect_stdout
from PIL import Image, ImageOps

# Set page title and layout
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")
st.title("MNIST Digit Recognition App")
st.write("This app demonstrates machine learning models for handwritten digit recognition using the MNIST dataset.")

# Define functions
@st.cache_data
def load_data(n_samples=10000):
    with st.spinner("Loading MNIST dataset..."):
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
        X = mnist["data"][:n_samples]
        y = mnist["target"][:n_samples].astype(np.uint8)
        return X, y

def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm).plot(ax=ax)
    return fig

def get_classification_report(y_true, y_pred):
    report_str = io.StringIO()
    with redirect_stdout(report_str):
        print(classification_report(y_true, y_pred, zero_division=0))
    return report_str.getvalue()

def preprocess_image(image):
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
   
    # Resize to 28x28
    image = image.resize((28, 28))
   
    # Invert colors (MNIST has white digits on black background)
    image = ImageOps.invert(image)
   
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
   
    # Reshape to match MNIST format (flatten)
    img_array = img_array.reshape(1, -1)
   
    return img_array

# Initialize session state for models if not exists
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = {}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Exploration", "Model Training", "Model Evaluation", "Predict Your Digit"])

# Load data
X, y = load_data()

if page == "Dataset Exploration":
    st.header("MNIST Dataset Exploration")
   
    col1, col2 = st.columns([1, 1])
   
    with col1:
        st.write(f"Dataset shape: {X.shape}")
        st.write(f"Number of classes: {len(np.unique(y))}")
       
        # Display sample distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        fig, ax = plt.subplots()
        ax.bar(unique_labels, counts)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of digits in dataset")
        st.pyplot(fig)
   
    with col2:
        # Display random samples
        st.subheader("Sample Images")
        n_samples = st.slider("Number of samples to display", 1, 25, 9)
       
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
       
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_samples)))
       
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
       
        # Handle both single axis and array of axes
        if n_samples == 1:
            img = X[sample_indices[0]].reshape(28, 28)
            axes.imshow(img, cmap=mpl.cm.binary)
            axes.set_title(f"Label: {y[sample_indices[0]]}")
            axes.axis('off')
        else:
            # Make sure axes is always 2D
            axes = np.array(axes).reshape(grid_size, grid_size)
           
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < n_samples:
                        img = X[sample_indices[idx]].reshape(28, 28)
                        axes[i, j].imshow(img, cmap=mpl.cm.binary)
                        axes[i, j].set_title(f"Label: {y[sample_indices[idx]]}")
                    axes[i, j].axis('off')
       
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Model Training":
    st.header("Model Training")
   
    # Data splitting options
    st.subheader("Data Splitting")
    test_size = st.slider("Test set size", 500, 3000, 2000, 100)
    val_size = st.slider("Validation set size", 500, 3000, 2000, 100)
   
    # Model selection
    st.subheader("Select Models to Train")
    use_logreg = st.checkbox("Logistic Regression", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    use_et = st.checkbox("Extra Trees", value=True)
    use_voting = st.checkbox("Voting Classifier", value=True)
   
    # Hyperparameters
    st.subheader("Hyperparameters")
   
    col1, col2 = st.columns(2)
   
    with col1:
        if use_logreg:
            st.write("Logistic Regression")
            logreg_max_iter = st.number_input("Max iterations", 100, 5000, 1000, 100)
       
        if use_rf:
            st.write("Random Forest")
            rf_n_estimators = st.number_input("Number of trees (RF)", 10, 500, 100, 10)
   
    with col2:
        if use_et:
            st.write("Extra Trees")
            et_n_estimators = st.number_input("Number of trees (ET)", 10, 500, 100, 10)
   
    # Train button
    if st.button("Train Models"):
        with st.spinner("Splitting data..."):
            # Splitting data
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, random_state=42)
           
            # Scaling data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            X_train_val = scaler.fit_transform(X_train_val)
           
            # Save scaler in session state
            st.session_state['scaler'] = scaler
            st.session_state['data'] = {
                'X_test': X_test,
                'y_test': y_test,
                'X_val': X_val,
                'y_val': y_val
            }
           
            # Initialize models
            models = {}
            model_names = []
           
            if use_logreg:
                logreg_clf = LogisticRegression(max_iter=logreg_max_iter)
                models["Logistic Regression"] = logreg_clf
                model_names.append("Logistic Regression")
           
            if use_rf:
                random_forest_clf = RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42)
                models["Random Forest"] = random_forest_clf
                model_names.append("Random Forest")
           
            if use_et:
                extra_trees_clf = ExtraTreesClassifier(n_estimators=et_n_estimators, random_state=42)
                models["Extra Trees"] = extra_trees_clf
                model_names.append("Extra Trees")
           
            named_estimators = []
            if use_logreg:
                named_estimators.append(("logreg_clf", logreg_clf))
            if use_rf:
                named_estimators.append(("random_forest_clf", random_forest_clf))
            if use_et:
                named_estimators.append(("extra_trees_clf", extra_trees_clf))
               
            if use_voting and len(named_estimators) >= 2:
                voting_clf = VotingClassifier(named_estimators, voting='hard')
                models["Voting Classifier"] = voting_clf
                model_names.append("Voting Classifier")
           
            # Train models
            progress_bar = st.progress(0)
            status_text = st.empty()
           
            for i, (name, model) in enumerate(models.items()):
                status_text.text(f"Training {name}...")
                model.fit(X_train, y_train)
                progress_bar.progress((i + 1) / len(models))
           
            # Save models in session state
            st.session_state['models'] = models
           
            # Evaluate models on validation set
            status_text.text("Evaluating models on validation set...")
           
            results = []
            for name, model in models.items():
                val_score = model.score(X_val, y_val)
                results.append({"Model": name, "Validation Accuracy": val_score})
           
            # Display results
            status_text.text("Training complete!")
            st.subheader("Model Performance on Validation Set")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
           
            # Plot bar chart of validation accuracies
            fig, ax = plt.subplots()
            ax.bar(results_df["Model"], results_df["Validation Accuracy"])
            ax.set_xlabel("Model")
            ax.set_ylabel("Validation Accuracy")
            ax.set_title("Model Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)


elif page == "Model Evaluation":
    st.header("Model Evaluation")
   
    if not st.session_state['models']:
        st.warning("No models have been trained yet. Please go to the 'Model Training' page first.")
    elif 'data' not in st.session_state:
        st.warning("No test data available. Please go to the 'Model Training' page first.")
    else:
        # Select model to evaluate
        model_names = list(st.session_state['models'].keys())
        selected_model = st.selectbox("Select model to evaluate", model_names)
       
        model = st.session_state['models'][selected_model]
        X_test = st.session_state['data']['X_test']
        y_test = st.session_state['data']['y_test']
       
        # Make predictions
        y_pred = model.predict(X_test)
       
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm_fig = display_confusion_matrix(y_test, y_pred)
        st.pyplot(cm_fig)
       
        # Display classification report
        st.subheader("Classification Report")
        report = get_classification_report(y_test, y_pred)
        st.text(report)
       
        # Display misclassified examples
        st.subheader("Misclassified Examples")
        misclassified = np.where(y_pred != y_test)[0]
       
        if len(misclassified) > 0:
            n_examples = min(10, len(misclassified))
            indices = np.random.choice(misclassified, n_examples, replace=False)
           
            cols = st.columns(5)
            for i, idx in enumerate(indices):
                col_idx = i % 5
                with cols[col_idx]:
                    # Get the scaled image and reshape it
                    img = X_test[idx].reshape(28, 28)
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(img, cmap=mpl.cm.binary)
                    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
                    ax.axis('off')
                    st.pyplot(fig)
        else:
            st.write("No misclassified examples found!")


elif page == "Predict Your Digit":
    st.header("Predict Your Digit")
   
    if not st.session_state['models']:
        st.warning("No models have been trained yet. Please go to the 'Model Training' page first.")
    else:
        st.write("Upload an image of a handwritten digit or draw one directly.")
       
        # Option to upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
       
        if uploaded_file is not None:
            # Read and display the image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
           
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, width=200)
           
            # Preprocess the image
            processed_img = preprocess_image(image)
           
            with col2:
                st.subheader("Preprocessed Image (28x28)")
                st.image(processed_img.reshape(28, 28), width=200)
           
            # Scale the image using the same scaler used for training
            if st.session_state['scaler'] is not None:
                scaled_img = st.session_state['scaler'].transform(processed_img)
            else:
                scaled_img = processed_img
           
            # Make predictions with all models
            st.subheader("Predictions")
           
            results = []
            for name, model in st.session_state['models'].items():
                prediction = model.predict(scaled_img)[0]
                probability = 1.0  # Default for models without predict_proba
               
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(scaled_img)[0][prediction]
               
                results.append({
                    "Model": name,
                    "Prediction": int(prediction),
                    "Confidence": probability
                })
           
            # Display results
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
           
            # Show the most common prediction
            most_common_pred = results_df["Prediction"].mode()[0]
            st.success(f"Most likely digit: {most_common_pred}")
           
            # Visualize predictions
            fig, ax = plt.subplots()
            ax.bar(results_df["Model"], results_df["Prediction"])
            ax.set_xlabel("Model")
            ax.set_ylabel("Predicted Digit")
            ax.set_title("Model Predictions")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
       
        st.write("Note: For best results, upload an image with a white digit on a black background, or a black digit on a white background.")
