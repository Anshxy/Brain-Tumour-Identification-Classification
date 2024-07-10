import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define parameters
img_height, img_width = 224, 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transformations
preprocess = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(image_file):
    """ Open and preprocess Image """
    image = Image.open(io.BytesIO(image_file)).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

def predict_binary(model, image):
    """
    Binary classification prediction
    If prob is greater than 0.5, return 'yes' (tumor present)
    If prob is less than 0.5, return 'no' (no tumor)
    """
    image = image.to(device)
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        if prob > 0.5:
            prediction = 'yes'
        else:
            prediction = 'no'
    
    return prediction, prob

def predict_tumor_type(model, image):
    """
    Multi-class classification prediction for tumour type
    0: Glioma, 1: Meningioma, 2: Pituitary
    """
    image = image.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Determine the class with the highest probability
        _, predicted_class = torch.max(probabilities, 0)
    
    return predicted_class.item(), probabilities.cpu().numpy().tolist()

# Load the pre-trained ResNet model for binary classification
binary_model = models.resnet18(pretrained=True)
num_ftrs = binary_model.fc.in_features

# Binary classification
binary_model.fc = nn.Linear(num_ftrs, 1)  
binary_model.load_state_dict(torch.load('Models/BTBinaryClassification.pth'))
binary_model = binary_model.to(device)

# Load the pre-trained ResNet model for multi-class tumor classification
tumor_model = models.resnet18(pretrained=True)
num_ftrs = tumor_model.fc.in_features

# Assuming three classes for tumor types
tumor_model.fc = nn.Linear(num_ftrs, 3)  
tumor_model.load_state_dict(torch.load('Models/BTMulticlassClassification.pth'))
tumor_model = tumor_model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image'].read()
    
    try:
        # Load and preprocess the image
        image = load_image(image_file)
        
        # Make the binary classification prediction
        binary_prediction, binary_probability = predict_binary(binary_model, image)
        
        if binary_prediction == 'yes':
            tumor_type, probabilities = predict_tumor_type(tumor_model, image)
            if tumor_type == 0:
                tumor_type = 'Glioma(0)'
            elif tumor_type == 1:    
                tumor_type = 'Meningioma(1)'
            else:
                tumor_type = 'Pituitary(2)'
                
            return jsonify({
                'binary_prediction': binary_prediction,
                'binary_probability': float(binary_probability),
                'tumor_type': tumor_type,
                'probabilities': probabilities
            })
        else:
            return jsonify({
                'binary_prediction': binary_prediction,
                'binary_probability': float(binary_probability)
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
