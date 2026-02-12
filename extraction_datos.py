import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import json
import pypdf
import os

def cargar_texto_desde_archivo(ruta_archivo):
    """Carga el texto de un archivo .txt o .pdf"""
    if not os.path.exists(ruta_archivo):
        return "Error: El archivo no existe."
    
    if ruta_archivo.lower().endswith('.pdf'):
        texto = ""
        try:
            reader = pypdf.PdfReader(ruta_archivo)
            for page in reader.pages:
                texto += page.extract_text() + "\n"
            return texto
        except Exception as e:
            return f"Error leyendo PDF: {e}"
    else:
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error leyendo archivo: {e}"

MI_CLAVE_GOOGLE = "AIzaSyAsqHzC0o5z72JCALqMfYKgIGRlSapgEFg" 

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=MI_CLAVE_GOOGLE  
)

nombre_archivo = "Whole-body Disorders that Affect the Skin in Dogs - Dog Owners - Merck Veterinary Manual.pdf" 
print(f"Leyendo archivo: {nombre_archivo}...")
texto_libro = cargar_texto_desde_archivo(nombre_archivo)

if len(texto_libro) < 100:
    print("ADVERTENCIA: El texto cargado parece muy corto o hubo una error.")
    print("Contenido:", texto_libro)

template = """
Eres un experto extrayendo datos sobre datos veterinario basicos que debe saber un dueño de un perro.
Analiza el siguiente texto y extrae 8 pares de preguntas y respuestas.

Sigue estrictamente este formato JSON (una lista de objetos):
[
  {{
    "pregunta": "Aquí va la pregunta generada...",
    "respuesta": "Aquí va la respuesta basada en el texto..."
  }},
  {{
    "pregunta": "Aquí va la segunda pregunta...",
    "respuesta": "Aquí va la segunda respuesta..."
  }}
]

Texto a analizar:
{texto}
"""

prompt = ChatPromptTemplate.from_template(template)

# --- PASO 6: EJECUCIÓN ---
print("Conectando con Google Gemini...")

try:
    mensaje = prompt.format_messages(texto=texto_libro)
    respuesta = chat_model.invoke(mensaje)
    
    # Parsear (limpiar) la respuesta manualmente
    contenido_limpio = respuesta.content.replace("```json", "").replace("```", "").strip()
    datos_limpios = json.loads(contenido_limpio)
    
    print("\n¡ÉXITO! Datos extraídos correctamente.")
    print("Guardando en dataset_veterinario_limpio.jsonl...\n")
    
    archivo_salida = "dataset_veterinario_limpio.jsonl"
    
    # Abrir en modo 'a' (append) para agregar al final sin borrar lo existente
    with open(archivo_salida, 'a', encoding='utf-8') as f:
        for i, dato in enumerate(datos_limpios, 1):
            # Crear la estructura requerida para formato chat
            nuevo_registro = {
                "messages": [
                    {"role": "user", "content": dato['pregunta']},
                    {"role": "model", "content": dato['respuesta']}
                ]
            }
            
            # Escribir en el archivo como una línea JSON válida
            f.write(json.dumps(nuevo_registro, ensure_ascii=False) + "\n")
            
            # Mostrar en consola para confirmación visual
            print(f"--- Registro {i} guardado ---")
            print(f"User: {dato['pregunta']}")
            print(f"Model: {dato['respuesta']}")
            print("")

    print(f"\nProceso finalizado. Se han agregado {len(datos_limpios)} nuevos registros al dataset.")

except Exception as e:
    print("\n--- OCURRIÓ UN ERROR ---")
    print(e)