# VisionQueryMaster: Advanced Visual Question Answering System

## Overview
Welcome to **VisionQueryMaster**, an advanced Visual Question Answering system that seamlessly integrates Optical Character Recognition (OCR) with deep contextual understanding using Qwen2-VL. This project goes beyond traditional OCR capabilities, allowing users to not only extract text from images in both English and Hindi but also to ask context-specific questions about the image content. The web-based application provides robust answers to questions ranging from color and object identification to summarization and contextual understanding, demonstrating cutting-edge AI capabilities.

This repository contains all the necessary files and code to run VisionQueryMaster locally or deploy it online. This project shows the immense potential of visual question-answering models combined with state-of-the-art NLP frameworks. 

On an average for a low quality image it takes 2-3 minutes for the model to answer any question regarding the image like a proper chatbot whereas for a high quality image with intensive text it might take 3-4 minutes to generate the answer but the results are amazing!

## Features
- **OCR with Dual Language Support:** Our system effectively extracts text from images in both **English and Hindi**, using a powerful pre-trained vision transformer model.
- **Contextual Question-Answering:** Ask questions about the image, and get meaningful, context-aware answers, such as identifying objects, describing scenes, or solving text-based queries.
- **Interactive Web Application:** The application is built using Gradio, providing a user-friendly interface for image uploads, text display, and question-asking functionality.
- **Dynamic Text Search:** Extracted text can be searched for specific keywords, making the app suitable for a variety of document analysis tasks.
- **Optimized Image Handling:** Images are resized dynamically to fit within memory limits, ensuring smooth performance even on limited resources.

## Try It Out
To try out the model, [**Click Here**](https://huggingface.co/spaces/vvinayakkk/seventhtry).

## Technical Details
### 1. **Model Selection and Implementation**
The initial phase of this project involved rigorous testing of multiple OCR models, including:
- **General OCR Theory (GOT)** - A lightweight 580M end-to-end OCR 2.0 model.
- **Tesseract and PaddleOCR** - Capable of text extraction but lacked context understanding.

After careful evaluation, we integrated the **ColPali implementation of the Qwen2-VL model**, leveraging Huggingface's Transformers library. This model not only excels at text extraction but also at image comprehension and question answering.

### 2. **Challenges Encountered and Solutions**
- **Memory Management:** The Qwen2-VL model, being a 2B parameter transformer model, initially occupied a staggering 10.2GB of RAM upon loading. This posed a major challenge when processing multiple images, as the memory would quickly exceed the 16GB limit on platforms like Hugging Face. By dynamically resizing images to a maximum resolution of 1000x1000 and effectively managing memory, we ensured stable performance.
- **Handling High-Resolution Images:** Images exceeding a certain resolution caused the application to crash due to excessive memory usage. We implemented a resizing mechanism to scale down high-resolution images while preserving their quality and information, thereby optimizing memory consumption.

### 3. **Web Application Development**
We created a highly interactive and responsive web application using **Streamlit**. The app consists of the following components:
- **Image Upload and Preview:** Users can upload images and view them directly in the app.
- **Text Extraction Display:** The extracted text is displayed in a dedicated section, giving users a quick glance at what was captured from the image.
- **Dynamic Q&A Section:** A sidebar offers a set of sample questions such as:
  1. "What is this image about?"
  2. "What is the color in the image?"
  3. "Summarize the image in 15 words."
  
  Users can also type in custom questions, and the system responds with accurate, context-aware answers.

### 4. **Deployment and Resource Management**
The web application has been successfully deployed on a HuggingFace via the streamlit spaces. Due to the high resource demands of the Qwen2-VL model, a GPU environment is recommended for optimal performance. However, the model has been tuned to run on CPU with limited capabilities for those lacking GPU access.

## Project Structure
The project directory is structured as follows:

```
.
├── app.py
├── requirements.txt
├── README.md
├── sample_images  # Folder containing sample images for testing
└── sample_questions.txt # Text file with sample questions for users

```


## Getting Started

### Prerequisites
Ensure that you have the following installed on your system:
- Python 3.7 or higher
- PyTorch 1.11 or higher
- Huggingface Transformers library
- Streamlit for web-based interface

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/VisionQueryMaster.git
   cd VisionQueryMaster

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. Run the Application To launch the web application locally, execute:
    ```bash
      
    streamlit run app.py

4. Ensure your system has at least 16GB of RAM for smooth execution. You can check available RAM by creating a sample file test_ram.py and running the file with the following code:
    ```bash 
       
    import psutil
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**2:.2f} MB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**2:.2f} MB")

## Deployment
For deployment on a cloud platform, ensure you have a GPU-enabled environment or modify the max_new_tokens parameter to 180 for CPU compatibility. Detailed instructions for deploying on platforms like Hugging Face and Streamlit can be found here. Link: https://huggingface.co/docs/hub/en/spaces-sdks-stream

## Sample Outputs and Screenshots
Here are a few examples showcasing the power of VisionQueryMaster:
 ![Screenshot 2024-09-30 212342](https://github.com/user-attachments/assets/df847db8-7306-4cd7-bab9-8770110385b2)

![Screenshot 2024-09-30 214302](https://github.com/user-attachments/assets/7f90e0fe-4623-496c-a46c-5ca8ca4dd94e)



## Results of Other Model Tried Out:
- **PaddleOCR**:![Screenshot 2024-09-30 203627](https://github.com/user-attachments/assets/38b68739-19d8-4ce6-a7c5-1a47a95a5278)

 - **Tesseract**:![Screenshot 2024-09-30 204416](https://github.com/user-attachments/assets/2da80d66-3dc3-48d8-b065-53cf357138c1)



## Future Enhancements
a) Live Webcam Capture: Integration of live webcam feed for real-time OCR and question answering.
b) Support for Additional Languages: Extend OCR capabilities to include more Indian regional languages.
c) Interactive Voice Q&A: Allow users to ask questions via speech and receive verbal responses, making the application accessible to a wider audience.

## Contact
For any questions or feedback, feel free to reach out via mail: vinayak.bhatia22@spit.ac.in






