import streamlit as st
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import time


@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return model, processor

model, processor = load_model()


st.title("Image Query App")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


st.sidebar.title("Suggested Questions")
predefined_questions = [
    "What is the main object in this image?",
    "Describe the scene in the image.",
    "Are there any people in the image?",
    "What is the background of the image?"
]
selected_question = st.sidebar.radio("Choose a question", predefined_questions)


question = st.sidebar.text_input("Or ask your own question here:")


submit_button = st.sidebar.button("Submit")


response = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    

    original_size = image.size
    st.write(f"Original image dimensions: {original_size}")


    max_size = (700, 700)
    if image.size[0] > 1000 or image.size[1] > 1000:
        image.thumbnail(max_size)
        resized_size = image.size
        st.write(f"Image resized to: {resized_size}")
    else:
        st.write("Image size is within acceptable limits.")

    if not question:
        question = selected_question


    if submit_button:
        st.sidebar.markdown("<h3 style='color:blue;'>Fetching the answer might take 2-3 minutes depending on the question, hold tight while we process your request!</h3>", unsafe_allow_html=True)
        start_time = time.time()  # Start the timer
        
        if question:
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            
            with st.spinner('Fetching the answer...'):
                with torch.no_grad():
                    new_generated_ids = model.generate(**inputs, max_new_tokens=180)

            
            new_generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, new_generated_ids)
            ]
            response = processor.batch_decode(
                new_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]  

        else:
            st.warning("Please enter a question.")

        elapsed_time = time.time() - start_time  # Calculate elapsed time


if response:
    st.markdown(f"<h4 style='color:green;'>Response:</h4><p style='font-size:18px;'>{response}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:gray;'>Time taken to fetch the answer: {elapsed_time:.2f} seconds</p>", unsafe_allow_html=True)


if uploaded_file is not None:
    st.image(image, caption='Uploaded Image', use_column_width=True)
