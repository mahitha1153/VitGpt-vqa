from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
import gradio as gr

# Load the model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def answer_question(image, text):
    # Convert the uploaded image to PIL format
    image = Image.fromarray(image.astype('uint8'), 'RGB')

    # Process the image and text
    encoding = processor(images=image, text=text, return_tensors="pt", padding=True)

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    idx = logits.argmax(-1).item()
    predicted_answer = model.config.id2label[idx]

    # Return the predicted answer
    return predicted_answer

# Define Gradio inputs and outputs
image = gr.Image(type="numpy", label="Upload Image")
question = gr.Textbox(lines=2, label="Question")
answer = gr.Textbox(label="Predicted Answer")

# Create Gradio Interface
gr.Interface(
    fn=answer_question, 
    inputs=[image, question], 
    outputs=answer,
    title="Image Based Visual Question Answering",
    description="This is a demonstration of ViLT (Vision and Language Transformer) using Gradio, which has been fine-tuned on VQAv2 to answer questions based on images. To get a predicted answer, please provide an image and type in your question, then press the submit button.\n\n DoneBy: BATCH-12 CSE-IoT"
).launch()
