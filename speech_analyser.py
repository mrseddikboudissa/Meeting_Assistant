import torch 
import os 
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM , pipeline, set_seed
#from langchain.llms import OpenAI from langchain.llms import HuggingFaceHub


#######------------- mistralai-------------####


model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
) 


#######------------- Speech2text-------------####

def transcript_audio(audio_file): # Initialize the speech recognition pipeline 
    pipe = pipeline( "automatic-speech-recognition", model="openai/whisper-tiny.en", chunk_length_s=30, ) 
    # Transcribe the audio file and return the result 
    transcript_txt = pipe(audio_file, batch_size=8)["text"] 
    temp = " List the key points with details from the floowing text context : \n {text} \n Key points:"

    prompt = temp.format(text=transcript_txt)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
        )

    # ✅ Only new tokens (important!)
    generated_tokens = output[0][inputs['input_ids'].shape[1]:]

    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result

#######------------- Gradio-------------####

audio_input = gr.Audio(sources="upload", type="filepath") 
output_text = gr.Textbox()

iface = gr.Interface(fn= transcript_audio, inputs= audio_input, outputs= output_text, title= "Audio Transcription App", description= "Upload the audio file")

iface.launch(server_name="0.0.0.0", server_port=7860)