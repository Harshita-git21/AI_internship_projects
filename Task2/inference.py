import tensorflow as tf
import numpy as np
from PIL import Image

# Define CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image_path, model_path='cifar10_mobilenetv2.keras'):
    """Loads a saved model and predicts the class of an image."""
    
    # 1. Load the saved model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model. Did you run the training script? Details: {e}")
        return

    # 2. Load and preprocess the image
    try:
        # Load image and resize to standard CIFAR-10 dimensions (32x32)
        # Note: The model handles the internal resizing to 96x96 and preprocessing
        img = Image.open(image_path).convert('RGB')
        img = img.resize((32, 32))
        
        # Convert to numpy array and add batch dimension: shape becomes (1, 32, 32, 3)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return

    # 3. Make the prediction
    predictions = model.predict(img_array, verbose=0)
    
    # 4. Interpret the results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = class_names[predicted_idx]
    
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    # Note: Replace 'sample_image.jpg' with an actual path to an image on your machine
    test_image = "OIP.jpg" 
    
    import os
    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print(f"Please provide a valid image path to test. '{test_image}' not found.")