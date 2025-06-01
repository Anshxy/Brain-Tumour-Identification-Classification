# NeuroVision - A Brain Tumour Identification & Classification Model

![image](https://github.com/Anshxy/Brain-Tumour-Identification-Classification/assets/96556167/b8d6b94b-41fd-49b7-9fb6-f3ca3880ad6f)

Hosted and deployed on [NeuroVision](https://braintumourdetect.vercel.app/)

![image](https://github.com/Anshxy/Brain-Tumour-Identification-Classification/assets/96556167/d89d4f12-916a-49af-a6c0-5f9b4f043384)

This project aims to identify and classify brain tumours from MRI images using deep learning models. The application includes both binary classification (to determine the presence of a tumour) and multi-class classification (to categorise the type of tumour: Glioma, Meningioma, or Pituitary).

## Getting started

1. Clone the Repository
```bash
git clone https://github.com/Anshxy/Brain-Tumour-Identification-Classification.git
cd Brain-Tumour-Identification-Classification
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Training your own model

The training data used in the pretrained models:
- [Multi-class classification](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Training)
- [Binary Classification](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri/data)

You can also train your own custom model by changing the training data pathing in the training files.

Binary Classification
```Python
# /trainers/Identify/training.ipynb

# Change training and validation path
train_dir = '../Identify/Data/training'
validation_dir = '../Identify/Data/validation' 
```

Multi-class Classification
```Python
# /trainers/Classification/training.ipynb

# Change training and validation path
train_dir = '../Classification/Data/training'
val_dir = f'../Classification/Data/validation'
```

Once done training, ensure the models are under the 'Models/' directory

## Testing and usage

1. Model preperation
Place your pre-trained models (BTBinaryClassification.pth and BTMulticlassClassification.pth) in the Models/ directory.

3. Start the Flask Server
```bash
python main.py
```
3. Open live server
Navigate to http://127.0.0.1:5000/ to access the web interface.

5. Upload and Predict
Use the web interface to upload MRI images and get predictions!

![image](https://github.com/Anshxy/Brain-Tumour-Identification-Classification/assets/96556167/010546df-2612-48ed-a310-0c679220708a)
- Note that this is a demo interface used for purely testing

## Author

By Ansh Rawat and Jun Oh
