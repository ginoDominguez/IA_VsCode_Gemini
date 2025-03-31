### instalar : pip install python-dotenv ### para guardar la clave de API
### pip install google-generativeai pillow   ### para pasar la imagen a formato pillow que lo entiende gemini.
import pathlib
import textwrap
from dotenv import load_dotenv
import os

import requests ### para descarga imagenes.
from PIL import Image   ### pasar a foramtro pillow
from io import BytesIO

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Used to securely store your API key   
#from google.colab import userdata

# Or use `os.getenv('GEMINI_API_KEY')` to fetch an environment variable.
#GOOGLE_API_KEY = userdata.get("tu clave") ### setx GOOGLE_API_KEY "" #guardada en clave de entorno

## Obetenr el key 
load_dotenv()  # Cargar variables del archivo .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("La API Key no está configurada. Asegúrate de haberla guardado como variable de entorno.")

genai.configure(api_key=GOOGLE_API_KEY)

###Use list_models to see the available Gemini models:
#gemini-1.5-flash: optimized for multi-modal use-cases where speed and cost are important. 
##This should be your go-to model.
#gemini-1.5-pro: optimized for high intelligence tasks, the most powerful Gemini model

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
        
#Always start with the 'gemini-1.5-flash' model. It should be sufficient for most of your tasks:
model = genai.GenerativeModel("gemini-1.5-flash")

#In the simplest case, you can pass a prompt string to the GenerativeModel.generate_content method:

response = model.generate_content("What is the meaning of life?")   
#to_markdown(response.text) ### Funciona solo en Jupyter

# Imprimir la respuesta en la terminal
print("\nRespuesta de Gemini:")
print(response.text)


##To stream responses, use GenerativeModel.generate_content(..., stream=True).
response = model.generate_content("What is the meaning of life?", stream=True)

for chunk in response:
    print(chunk.text)
    print("_" * 80)

    
   
url="https://t0.gstatic.com/licensed-image?q=tbn:ANd9GcQ_Kevbk21QBRy-PgB4kQpS79brbmmEG7m3VOTShAn4PecDU5H5UxrJxE3Dw1JiaG17V88QIol19-3TM2wCHw" 

# Descargar la imagen
response = requests.get(url)

# Guardar la imagen en un archivo
if response.status_code == 200:
    img= Image.open(BytesIO(response.content))  ### convierto la imagen
    with open("image.jpg", "wb") as file:
        file.write(response.content)
    print("✅ Imagen descargada como 'image.jpg'")
else:
    print("❌ Error al descargar la imagen")
    
"""    
model = genai.GenerativeModel("gemini-1.5-flash")

# Generar contenido con la imagen
response = model.generate_content([img, "Describe this image."])

# Imprimir respuesta
print("\n🖼️ Respuesta de Gemini:")
print(response.text)


#To provide both text and images in a prompt, pass a list containing the strings and images:
# Generar contenido con la imagen y un prompt para el blog post
response = model.generate_content(
    [
        "Write a short, engaging blog post based on this picture. It should include a description of the meal in the photo and talk about my journey meal prepping.",
        img,
    ],
    stream=True,  # Habilitar streaming para recibir la respuesta en tiempo real
)

# Resolver la respuesta
response.resolve()

# Imprimir el texto generado
print("\n📝 Blog Post generado por Gemini:")
print(response.text)
"""


#### Chat Conversattion:


model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])
##chat


response = chat.send_message(
    "In one sentence, explain how a computer works to a young child."
)
print(response.text)

print("\n💬 Historial del Chat:")
for msg in chat.history:
    print(f"{msg.role}: {msg.parts}")

response = chat.send_message(
    "Okay, how about a more detailed explanation to a high schooler?", stream=True
)

for chunk in response:
    print(chunk.text)
    print("_" * 80)
    

for message in chat.history:
    print(f"**{message.role}**: {message.parts[0].text}")
    
### COUNT TOKENS:
print(model.count_tokens("What is the meaning of life?"))

print(model.count_tokens(chat.history))


#### Advanced use of cases

result = genai.embed_content(
    model="models/text-embedding-004",
    content="What is the meaning of life?",
    task_type="retrieval_document",
    title="Embedding of single string",
)

# 1 input > 1 vector output
print(str(result["embedding"])[:50], "... TRIMMED]")

###The retrieval_document task type is the only task that accepts a title.

result = genai.embed_content(
    model="models/text-embedding-004",
    content=[
        "What is the meaning of life?",
        "How much wood would a woodchuck chuck?",
        "How does the brain work?",
    ],
    task_type="retrieval_document",
    title="Embedding of list of strings",
)

# A list of inputs > A list of vectors output
for v in result["embedding"]:
    print(str(v)[:50], "... TRIMMED ...")


response.candidates[0].content

result = genai.embed_content(
    model="models/text-embedding-004", content=response.candidates[0].content
)

# 1 input > 1 vector output
print(str(result["embedding"])[:50], "... TRIMMED ...")
print("Chat History \n")

print(chat.history)

result = genai.embed_content(model="models/text-embedding-004", content=chat.history)

# 1 input > 1 vector output
for i, v in enumerate(result["embedding"]):
    print(str(v)[:50], "... TRIMMED...")
    
    
    
###Safety Settings:
print("Safety Settings \n")
response = model.generate_content("[Questionable prompt here]")
print(response.candidates)

print(response.prompt_feedback)

#### changin the parameter harassment:
response = model.generate_content(
    "[Questionable prompt here]", safety_settings={"HARASSMENT": "block_none"}
)
print(response.text)


model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    genai.protos.Content(
        parts=[
            genai.protos.Part(
                text="Write a short, engaging blog post based on this picture."
            ),
            genai.protos.Part(
                inline_data=genai.protos.Blob(
                    mime_type="image/jpeg", data=pathlib.Path("image.jpg").read_bytes()
                )
            ),
        ],
    ),
    stream=True,
)

response.resolve()

print(response.text[:100] + "... [TRIMMED] ...")


### Multi Turn conversation:

model = genai.GenerativeModel("gemini-1.5-flash")

messages = [
    {
        "role": "user",
        "parts": ["Briefly explain how a computer works to a young child."],
    }
]
response = model.generate_content(messages)

print(response.text)

### To continue the conversation, add the response and another message.
messages.append({"role": "model", "parts": [response.text]})

messages.append(
    {
        "role": "user",
        "parts": [
            "Okay, how about a more detailed explanation to a high school student?"
        ],
    }
)

response = model.generate_content(messages)

print(response.text)

#### Generation configuration

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    "Tell me a story about a magic backpack.",
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        stop_sequences=["x"],
        max_output_tokens=20,
        temperature=1.0,
    ),
)


text = response.text

if response.candidates[0].finish_reason.name == "MAX_TOKENS":
    text += "..."

print(text)
