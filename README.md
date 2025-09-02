🌱 Crop Disease Detection System

An AI-powered web application that detects plant diseases from leaf images using Deep Learning.
The system provides real-time predictions, helping farmers and researchers identify crop health issues early and take preventive measures.

🚀 Features

Disease Detection
Upload an image of a crop leaf and get instant predictions with confidence scores.

Wide Crop Coverage
Supports multiple plants including Apple, Corn, Grape, Potato, Tomato, and more.

Deep Learning Model
Built using TensorFlow/Keras CNN trained on the PlantVillage dataset.

Interactive Web App
Developed with Streamlit for a user-friendly, responsive interface.

Training Insights
Includes training history (training_hist.json) with accuracy and loss metrics.

📊 Model Performance

Training Accuracy: 98%

Validation Accuracy: 97%

Validation Loss: 0.17

📈 Accuracy improves steadily across epochs, with minimal overfitting.

📂 Project Structure
├── main.py                 # Streamlit web app for prediction
├── Train.ipynb             # Jupyter Notebook for training the CNN model
├── Test.ipynb              # Jupyter Notebook for testing/evaluating the model
├── training_hist.json      # Training history (accuracy, loss, val_accuracy, val_loss)
└── README.md               # Project documentation

⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/crop-disease-detection.git
cd crop-disease-detection


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

▶️ Usage

Train the model (if not already trained):

Open and run Train.ipynb to train the CNN model.

The trained model will be saved as trained_plant_disease_model.keras.

Run the app:

streamlit run main.py


Use the Web App:

Go to the Disease Detection page.

Upload a leaf image (jpg/png).

Click Predict to see the detected disease and confidence score.

🧑‍💻 Tech Stack

Frontend/UI: Streamlit

Deep Learning: TensorFlow, Keras

Data Handling: NumPy, Pandas

Image Processing: PIL

Visualization: Matplotlib (for training insights)

🌾 Supported Classes

Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy

Corn (Maize): Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy

Grape: Black Rot, Esca, Leaf Blight, Healthy

Potato: Early Blight, Late Blight, Healthy

Tomato: Bacterial Spot, Early/Late Blight, Leaf Mold, Septoria, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

Others: Blueberry, Cherry, Orange, Peach, Pepper, Raspberry, Soybean, Squash, Strawberry


🌟 Acknowledgements

PlantVillage Dataset

TensorFlow/Keras for deep learning

Streamlit for interactive app deployment
