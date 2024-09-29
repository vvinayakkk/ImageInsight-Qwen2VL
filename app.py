import streamlit as st
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    return model, processor

model, processor = load_model()

st.title("Image Query App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
   
    original_size = image.size
    st.write(f"Original image dimensions: {original_size}")

    
    max_size = (700, 700)
    if image.size[0] > 1000 or image.size[1] > 1000:
        image.thumbnail(max_size)
        resized_size = image.size
        st.write(f"Image resized to: {resized_size}")
    else:
        st.write("Image size is within acceptable limits.")

    question = st.text_input("What do you want to know about this image?")

    if st.button("Submit"):
        if question:
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
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
            decoded_text = processor.batch_decode(
                new_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            response = decoded_text[0] 
            st.write("Response:", response)  
        else:
            st.warning("Please enter a question.")
