# AI-Based Skin Cancer Detection System

## Overview
The AI-based skin cancer detection system is a deep learning model designed to classify four types of skin cancer. It uses TensorFlow to analyze skin lesion images and determine the cancer type, providing a non-invasive and efficient diagnostic tool. The model is deployed on a Raspberry Pi 3 for real-time classification.

## Features
- **Multi-class Skin Cancer Detection:** Classifies skin lesions into four categories.
- **Real-time Image Analysis:** Processes images from a connected camera.
- **Raspberry Pi 3 Deployment:** Runs efficiently on an embedded system.
- **User-friendly Interface:** Displays classification results and confidence scores.
- **TensorFlow Lite Optimization:** Ensures fast and accurate inference.

## Hardware Requirements
- Raspberry Pi 3 Model B/B+
- Raspberry Pi Camera Module
- LCD Display (optional for visual feedback)
- Battery Pack (suitable for Raspberry Pi)
- SD Card (minimum 16GB, recommended 32GB)

## Software Requirements
- Python 3.7+
- TensorFlow (Lite version for Raspberry Pi)
- OpenCV
- NumPy
- Matplotlib (for visualization)
- Picamera (for capturing images)

## Model Training & Deployment
### Model Training
- The skin cancer detection model was trained on a dataset containing four classes of skin cancer.
- A Convolutional Neural Network (CNN) architecture was used.
- TensorFlow was used to train and optimize the model for real-time inference.

### Deployment on Raspberry Pi
- The trained model was converted to TensorFlow Lite format for efficient execution.
- Image processing and inference were handled using OpenCV and TensorFlow Lite.
- The model was integrated with an LCD display to show classification results.

## Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/skin-cancer-detector.git
   cd skin-cancer-detector
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python numpy matplotlib picamera
   ```
3. Run the detection script:
   ```bash
   python detect_skin_cancer.py
   ```

## Usage Instructions
- Ensure the camera is properly positioned for capturing images of skin lesions.
- Run the script to start real-time skin cancer classification.
- The system will display the classification results and confidence scores.

## Example Output
- Example images of detected skin cancer types with predicted labels.
- Log output showing detection confidence and classification results.

## Troubleshooting
- **Model not loading?** Ensure TensorFlow Lite is installed correctly.
- **Camera not working?** Check if `picamera` is enabled on Raspberry Pi.
- **Low accuracy?** Ensure the dataset is well-prepared and balanced.

## Contributors & Credits
- Developed by Yahya Abdurrazaq
- Special thanks to open-source datasets and TensorFlow community

## License
This project is licensed under the MIT License.

