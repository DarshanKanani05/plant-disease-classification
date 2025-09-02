# 🌱 Plant Disease Classification with EfficientNetB0

## 📌 Project Overview
This project builds a deep learning model to classify plant leaf images into multiple disease categories. Early detection of plant diseases is crucial for improving crop yield and preventing losses in agriculture.  

Using **EfficientNetB0**, a state-of-the-art CNN architecture, the model achieves high accuracy on the **New Plant Diseases Dataset (Augmented)** from Kaggle.

---

## 📊 Dataset
- **Source**: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- **Classes**: 38 different plant diseases + healthy leaves  
- **Split**: Provided train/validation/test folders  

---

## ⚙️ Tech Stack
- **Language**: Python  
- **Frameworks**: PyTorch, Torchvision  
- **Other tools**: Pandas, NumPy, Matplotlib, KaggleHub, Google Colab  

---

## 🚀 Approach
1. **Data Preprocessing**
   - Image resizing to 224×224  
   - Normalization (ImageNet mean/std)  
   - Train/validation/test split  

2. **Model**
   - EfficientNetB0 (transfer learning)  
   - Adam optimizer, learning rate scheduling  
   - Gradient clipping + weight decay  

3. **Training**
   - Batch size = 32, Epochs = 10  
   - Trained on GPU (Google Colab)  

4. **Evaluation**
   - Accuracy and loss tracking  
   - Visualization of predictions  

5. **Deployment**
   - Trained model saved as `.pth`  
   - Inference script with UI for testing on new images  

---

## 📈 Results
- Achieved strong classification accuracy on validation & test sets.  
- Visualizations show good generalization across disease categories.  

*(Add final accuracy/metrics once you confirm from notebook outputs.)*  

---

## 📂 Project Structure
```
├── notebooks/
│   └── PlantDisease_EfficientNet.ipynb
├── data/               # Kaggle dataset (not included here)
├── saved_models/       # Trained .pth file
├── images/             # Sample predictions & charts
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run
1. Clone this repo:  
   ```bash
   git clone https://github.com/DarshanKanani05/plant-disease-classification.git
   cd plant-disease-classification
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Download dataset from Kaggle and place it under `data/`.  
4. Run the notebook in Jupyter/Colab to train or use the inference script to test with new images.  

---

## 🌍 Future Work
- Expand to larger datasets  
- Optimize inference for mobile/edge deployment  
- Deploy model with a web dashboard or API  

---

## 🙌 Acknowledgements
- Dataset: [Vipoooool – Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- EfficientNetB0 paper: *Tan & Le (2019)*  
