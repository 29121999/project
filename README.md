Lung Disease Prediction Using Machine Learning Algorithms
Project Overview
This project leverages machine learning techniques to predict lung diseases, specifically Pneumonia and Tuberculosis, using chest X-ray images. The aim is to develop robust models that assist in rapid and accurate disease detection to improve healthcare outcomes.
Key Objectives:
•	Apply preprocessing and exploratory data analysis (EDA) to medical image datasets.
•	Train and evaluate various deep learning models:
o	Custom Convolutional Neural Networks (CNNs)
o	Pre-trained architectures: VGG16, DenseNet121, and InceptionResNetV2
•	Address class imbalance using techniques like data augmentation and resampling.
•	Optimize model performance through hyperparameter tuning.
________________________________________
Datasets
1. Pneumonia Dataset:
•	Source: Guangzhou Women and Children’s Medical Center.
•	Images: 5,856 chest X-rays (JPEG format).
•	Categories:
o	Normal: 1,583 images
o	Bacterial Pneumonia: 2,780 images
o	Viral Pneumonia: 1,493 images
2. Tuberculosis Dataset:
•	Source: University of Dhaka and Qatar University collaboration.
•	Images: 4,200 chest X-rays.
•	Categories:
o	Normal: 3,500 images
o	TB-positive: 700 images
________________________________________
Workflow
Pneumonia Dataset
1.	Baseline Model (Imbalanced Data):
o	Train CNN from scratch and evaluate its performance.
2.	Hyperparameter Tuning:
o	Optimize CNN using techniques like early stopping and learning rate adjustment.
3.	Pre-trained Architectures:
o	Fine-tune VGG16, DenseNet121, and other models for enhanced accuracy.
4.	Data Augmentation:
o	Rotate, flip, zoom, and shift images to mitigate class imbalance.
Tuberculosis Dataset
1.	Custom CNN Model:
o	Train a sequential CNN architecture.
2.	Transfer Learning:
o	Use InceptionResNetV2 for binary classification.
3.	Class Balancing:
o	Apply oversampling (SMOTE) to balance Normal and TB-positive categories.
Combined Dataset
•	Integrate Pneumonia and Tuberculosis datasets to train a multi-disease classifier using VGG16.
________________________________________
Evaluation Metrics
•	Accuracy: Overall classification success rate.
•	Precision: Correct positive predictions over total positive predictions.
•	Recall: Correct positive predictions over total actual positives.
•	F1 Score: Harmonic mean of precision and recall.
•	Confusion Matrix: Breakdown of true/false positives and negatives.
•	ROC-AUC: Evaluates the trade-off between sensitivity and specificity.
________________________________________
Model Performance
Pneumonia Dataset
Model	Accuracy	AUC	Confusion Matrix	Comments
Custom CNN	70.19%	0.76	61 TP, 75 FP, 62 FN	Needs improvement with more data balancing.
VGG16	96.7%	0.99	88 TP, 2 FP, 1 FN	Excellent performance with fine-tuning.
DenseNet121	84.7%	0.94	86 TP, 5 FP, 3 FN	Robust model for classification tasks.
Tuberculosis Dataset
Model	Accuracy	AUC	Confusion Matrix	Comments
Custom CNN	93.3%	0.97	95 TP, 3 FP, 5 FN	Strong performance with custom architecture.
InceptionResNetV2	97.6%	0.99	98 TP, 2 FP, 1 FN	Near-perfect classification accuracy.
Combined Dataset
Model	Accuracy	AUC	Confusion Matrix	Comments
VGG16	89%	0.96	85 TP, 3 FP, 2 FN	Balanced performance for multi-disease classification.
________________________________________
Requirements
•	Python 3.9
•	Core Libraries: numpy, pandas, matplotlib, seaborn
•	Machine Learning: tensorflow, keras, scikit-learn
•	Data Augmentation: imgaug
•	Class Balancing: imbalanced-learn
•	Model Evaluation: roc_auc_score, classification_report
________________________________________
How to Run
1.	Clone the repository:
bash
Copy code
git clone https://github.com/your-username/LungDiseasePrediction.git
cd LungDiseasePrediction
2.	Install dependencies:
bash
Copy code
pip install -r requirements.txt
3.	Prepare datasets:
o	Place Pneumonia and Tuberculosis datasets in the data/ folder.
4.	Train models:
bash
Copy code
python scripts/train.py --model VGG16 --dataset combined
5.	Evaluate models:
bash
Copy code
python scripts/evaluate.py --model InceptionResNetV2
________________________________________
Future Work
1.	Expand Dataset: Incorporate additional lung conditions like COPD or lung cancer.
2.	Improve Model Efficiency: Develop lightweight versions for deployment on edge devices.
3.	Advanced Augmentation: Use GANs to generate synthetic X-ray images for balancing.
4.	Explainable AI: Integrate tools like Grad-CAM to visualize model predictions.
________________________________________
Acknowledgments
•	Pneumonia Dataset: Guangzhou Women and Children’s Medical Center.
•	Tuberculosis Dataset: University of Dhaka and Qatar University collaboration.
•	Pre-trained model architectures sourced from TensorFlow and Keras.

