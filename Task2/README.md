# CIFAR-10 Image Classification with MobileNetV2

This project demonstrates training a MobileNetV2 model for image classification on the CIFAR-10 dataset. It includes a Jupyter Notebook for training and a Python script for inference.

## Folder Structure

```
.
├── cifar10_mobilenetv2.keras
├── inference.py
├── OIP.jpg
├── requirements.txt
├── train.ipynb
├── training_curves.png
└── venv/
```

- `cifar10_mobilenetv2.keras`: The trained Keras model.
- `inference.py`: Script to run inference on a single image.
- `OIP.jpg`: An example image for inference.
- `requirements.txt`: Python dependencies.
- `train.ipynb`: Jupyter Notebook for training the model.
- `training_curves.png`: Plot of training and validation accuracy and loss.

## Environment Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**

    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install the required dependencies:**

    To get rid of errors, ensure all the necessary modules are installed. Run the following command:
    ```bash
    pip install -r requirements.txt
    ```
    This will install the following required modules:
    - tensorflow
    - matplotlib
    - numpy
    - pillow
    - jupyter

## Running the code

### Training

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  Open and run the `train.ipynb` notebook.

### Inference

To run inference on an image, use the `inference.py` script:

```bash
python inference.py --image_path OIP.jpg
```
