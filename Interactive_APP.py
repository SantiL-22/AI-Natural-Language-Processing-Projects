import streamlit as st
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import openai
from openai import OpenAI


OPENAI_ORG = ""
OPENAI_KEY = ""
client = OpenAI(api_key=OPENAI_KEY)
# -----------------------------
# IMAGE DESCRIPTION FUNCTION (BLIP-like)
# -----------------------------
# Placeholder function simulating a BLIP-base image captioning model.
# Replace with actual BLIP model inference if needed.
def describe_image(image: Image.Image):

    # Cargar modelo y procesador BLIP-base
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Asegurarnos de que la imagen esté en RGB
    image = image.convert("RGB")

    # Preparar inputs para el modelo
    inputs = processor(images=image, return_tensors="pt")

    # Generar la descripción
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)

    # Decodificar la salida a texto
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Retornar la descripción
    return caption

# -----------------------------
# STORY GENERATION FUNCTION
# -----------------------------
def generate_story(description, OPENAI_ORG, OPENAI_KEY):
    openai.organization = OPENAI_ORG
    openai.api_key = OPENAI_KEY
    
    response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un escritor creativo."},#Como se comporta el modelo
            {"role": "user", "content": f"Crea una historia corta basada en esta descripción en inlges de no mas de 100 palabras: {description}"}
        ])
    story = response.choices[0].message.content
    return story

# -----------------------------
# TEXT-TO-SPEECH FUNCTION (Kokoro-style)
# -----------------------------
def story_to_audio(story_text, OPENAI_ORG, OPENAI_KEY):
    # Uso de Kokoro TTS local con KPipeline
    from kokoro import KPipeline
    import soundfile as sf
    import torch

    pipeline = KPipeline(lang_code='a')
    generator = pipeline(story_text, voice='af_heart')

    # Unimos los fragmentos generados
    full_audio = []
    for _, _, audio in generator:
        full_audio.extend(audio)

    # Convertimos a bytes WAV
    import io
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, 24000, format='WAV')
    audio_bytes = buffer.getvalue()

    return audio_bytes

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Generador de Historias a partir de Imágenes", layout="wide")

st.title("🖼️ ➡️ 📖 ➡️ 🔊 Generador de Historias con Audio 🖼️ ➡️ 📖 ➡️ 🔊")
st.markdown("Sube una imagen, deja que la IA la describa, invente una historia y la convierta en audio.")

st.sidebar.markdown("---")
option = st.sidebar.radio("Selecciona método para cargar imagen", ["Subir archivo", "Introducir URL"])

image = None
if option == "Subir archivo":
    uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
elif option == "Introducir URL":
    url = st.text_input("Introduce el URL de la imagen")
    if url:
        try:
            resp = requests.get(url)
            image = Image.open(BytesIO(resp.content))
        except:
            st.error("No se pudo cargar la imagen desde el URL.")

if image:
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Generar descripción ➜ historia ➜ audio"):
        if not OPENAI_KEY or not OPENAI_ORG:
            st.error("Necesitas introducir tus credenciales de OpenAI en la barra lateral.")
        else:
            with st.spinner("Generando descripción..."):
                description = describe_image(image)

            st.subheader("📷 Descripción generada por el modelo:")
            st.write(description)

            with st.spinner("Escribiendo historia creativa..."):
                story = generate_story(description, OPENAI_ORG, OPENAI_KEY)

            st.subheader("📖 Historia generada:")
            st.write(story)

            with st.spinner("Convirtiendo historia en audio..."):
                audio = story_to_audio(story,OPENAI_ORG , OPENAI_KEY)

            st.subheader("🔊 Escuchar historia:")
            st.audio(audio)
