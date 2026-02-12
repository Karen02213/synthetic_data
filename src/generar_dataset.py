import os
import sys
import json
import time
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# CONFIGURACIÓN
# ============================================================
MI_CLAVE_GOOGLE = "AIzaSyAsqHzC0o5z72JCALqMfYKgIGRlSapgEFg"

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'refined_text')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

# Archivos de salida
SALIDA_GENERICO = os.path.join(DATASETS_DIR, 'dataset_veterinario_limpio.jsonl')
SALIDA_GEMINI = os.path.join(DATASETS_DIR, 'dataset_veterinario_gemini.jsonl')
ARCHIVO_LOG = os.path.join(DATASETS_DIR, 'processed_log.txt')

# Control de flujo
MAX_REINTENTOS = 3
PAUSA_ENTRE_ARCHIVOS = 2
PAUSA_REINTENTO = 5
MIN_CHARS_TEXTO = 50
MIN_CHARS_MENSAJE = 15
MIN_TURNOS = 2  # Mínimo 1 preguntas + 1 respuestas = 2 turnos

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.5,
    google_api_key=MI_CLAVE_GOOGLE
)

# ============================================================
# PROMPT — Conversaciones multi-turno
# ============================================================
TEMPLATE = """Eres un veterinario experto. Tu tarea es generar conversaciones realistas entre un dueño de mascota y un asistente veterinario IA llamado GoodPawies, basándote en el texto proporcionado.

Cada conversación debe simular una consulta real donde el dueño hace una pregunta, recibe respuesta, y eso lo lleva a hacer preguntas de seguimiento naturales. Es una conversación fluida, no preguntas aisladas.

=== PERSONALIDAD DEL ASISTENTE (role: "model") ===
El asistente se llama GoodPawies y tiene estas características:
- SIEMPRE responde en español latinoamericano.
- Es compasivo y comprensivo: los dueños suelen estar preocupados por sus mascotas.
- Tono amigable y cercano durante toda la conversación.
- Usa términos sencillos, evita jerga médica a menos que la explique de forma simple.
- SOLO responde sobre perros y gatos. Si el texto trata de otro animal, adapta la conversación a perros o gatos.
- NO da diagnósticos definitivos. Ofrece orientación general e información.
- Hace preguntas aclaratorias cuando es necesario (edad, raza, duración de síntomas, etc.).
- Da consejos prácticos de primeros auxilios cuando aplique, pero enfatiza la atención profesional.
- Puede recomendar medicamentos recetados específicos, pero menciona que solo un veterinario puede recetar.
- SIEMPRE recuerda al usuario que sus consejos no reemplazan la atención veterinaria profesional.
- Si los síntomas sugieren emergencia (dificultad para respirar, convulsiones, sangrado severo, ingestión de tóxicos), insta a buscar atención veterinaria inmediata.
- Las respuestas deben ser concisas pero informativas.

=== REGLAS DE GENERACIÓN ===
1. Genera minimo 6 conversaciones distintas a partir del texto.
2. Cada conversación debe tener MÍNIMO 1 intercambio (pregunta-respuesta), pueden ser más.
3. Las preguntas del usuario deben sonar naturales: "Mi perro tiene...", "¿Y qué pasa si...?", "¿Entonces debería...?"
4. Las respuestas del modelo deben ser profesionales pero accesibles, basadas ESTRICTAMENTE en el texto proporcionado.
5. NO inventes datos, medicamentos ni tratamientos que no estén en el texto.
6. Varía los temas: síntomas, causas, tratamientos, prevención, cuidados.
7. Los roles son siempre "user" y "model" (no "assistant").
8. Responde ÚNICAMENTE con un array JSON válido, sin texto adicional ni bloques de código.
9. Cada respuesta del modelo debe contener al menos 2-3 oraciones para ser útil.
10. La primera respuesta del modelo en cada conversación debe iniciar con un saludo como "¡Hola! Soy GoodPawies, tu asistente veterinario." o variaciones naturales.
11. Si el texto lo sugiere, incluir un consejo final de "consultar al veterinario".
12. Si el texto menciona síntomas o condiciones, asegúrate de que las preguntas y respuestas aborden esos puntos.
13. El asistente puede hacer preguntas de seguimiento al dueño para obtener más contexto antes de dar consejos.
=== FORMATO DE SALIDA ===
Responde ÚNICAMENTE con un array JSON válido. Sin texto extra, sin bloques de código.
Cada elemento es una conversación con un array de mensajes alternando user/model:

[
  {{
    "messages": [
      {{"role": "user", "content": "Primera pregunta del dueño"}},
      {{"role": "model", "content": "Respuesta del veterinario"}},
      {{"role": "user", "content": "Pregunta de seguimiento"}},
      {{"role": "model", "content": "Respuesta de seguimiento"}}
    ]
  }},
  {{
    "messages": [
      {{"role": "user", "content": "Otra consulta diferente"}},
      {{"role": "model", "content": "Respuesta"}},
      {{"role": "user", "content": "Seguimiento"}},
      {{"role": "model", "content": "Respuesta final"}}
    ]
  }}
]

=== TEXTO FUENTE ===
{texto}"""

prompt_template = ChatPromptTemplate.from_template(TEMPLATE)


# ============================================================
# UTILIDADES
# ============================================================

def obtener_texto_seguro(respuesta):
    """Extrae texto de la respuesta IA (str o list)."""
    content = respuesta.content
    if isinstance(content, list):
        partes = []
        for c in content:
            if isinstance(c, str):
                partes.append(c)
            elif isinstance(c, dict) and "text" in c:
                partes.append(c["text"])
            elif hasattr(c, "text"):
                partes.append(c.text)
            else:
                partes.append(str(c))
        return "".join(partes)
    if isinstance(content, str):
        return content
    return str(content)


def parsear_json(contenido):
    """Extrae y parsea JSON de la respuesta."""
    contenido = contenido.strip()
    contenido = re.sub(r'^```(?:json)?\s*', '', contenido)
    contenido = re.sub(r'\s*```$', '', contenido)
    contenido = contenido.strip()

    try:
        return json.loads(contenido)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[\s*\{.*\}\s*\]', contenido, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validar_conversacion(conv):
    """Valida una conversación: debe tener al menos MIN_TURNOS mensajes alternando user/model."""
    if not isinstance(conv, dict):
        return None
    
    messages = conv.get("messages", [])
    if not isinstance(messages, list) or len(messages) < MIN_TURNOS:
        return None

    mensajes_validos = []
    rol_esperado = "user"

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        rol = msg.get("role", "").strip()
        content = msg.get("content", "").strip()

        # Normalizar rol
        if rol == "assistant":
            rol = "model"

        if rol != rol_esperado:
            continue
        if len(content) < MIN_CHARS_MENSAJE:
            continue

        mensajes_validos.append({"role": rol, "content": content})
        rol_esperado = "model" if rol == "user" else "user"

    # Debe terminar en model y tener al menos MIN_TURNOS
    if len(mensajes_validos) < MIN_TURNOS:
        return None
    # Asegurar que termina en model
    if mensajes_validos[-1]["role"] != "model":
        mensajes_validos = mensajes_validos[:-1]
    if len(mensajes_validos) < MIN_TURNOS:
        return None

    return {"messages": mensajes_validos}


def conv_a_gemini(conv):
    """Convierte formato genérico a formato Gemini native."""
    contents = []
    for msg in conv["messages"]:
        contents.append({
            "role": msg["role"],
            "parts": [{"text": msg["content"]}]
        })
    return {"contents": contents}


# ============================================================
# PROCESAMIENTO
# ============================================================

def procesar_archivo(ruta):
    """Procesa un archivo y devuelve lista de conversaciones validadas."""
    nombre = os.path.basename(ruta)

    try:
        with open(ruta, 'r', encoding='utf-8') as f:
            texto = f.read()
    except Exception as e:
        print(f"  [ERROR] No se pudo leer: {e}")
        return []

    if len(texto.strip()) < MIN_CHARS_TEXTO:
        print(f"  [SKIP] Texto demasiado corto ({len(texto.strip())} chars)")
        return []

    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            mensaje = prompt_template.format_messages(texto=texto)
            respuesta = chat_model.invoke(mensaje)
            raw = obtener_texto_seguro(respuesta)

            datos = parsear_json(raw)
            if datos is None:
                print(f"  [WARN] Intento {intento}: JSON inválido, reintentando...")
                time.sleep(PAUSA_REINTENTO)
                continue

            if not isinstance(datos, list):
                datos = [datos]

            conversaciones = []
            for conv in datos:
                validada = validar_conversacion(conv)
                if validada:
                    conversaciones.append(validada)

            if not conversaciones:
                print(f"  [WARN] Intento {intento}: Ninguna conversación válida, reintentando...")
                time.sleep(PAUSA_REINTENTO)
                continue

            total_turnos = sum(len(c["messages"]) for c in conversaciones)
            print(f"  [OK] {len(conversaciones)} conversaciones, {total_turnos} turnos totales")
            return conversaciones

        except Exception as e:
            print(f"  [ERROR] Intento {intento}/{MAX_REINTENTOS}: {e}")
            if intento < MAX_REINTENTOS:
                time.sleep(PAUSA_REINTENTO)

    print(f"  [FAIL] No se pudo procesar después de {MAX_REINTENTOS} intentos")
    return []



def reconstruct_log_from_dataset(archivos_todos, log_path, dataset_path):
    """
    Intenta reconstruir el log de procesados escaneando el dataset existente.
    """
    print("  [INFO] Log no encontrado o vacío pero dataset existe. Intentando reconstruir estado...")
    
    if not os.path.exists(dataset_path):
        return set()

    processed = set()
    try:
        # Leer dataset para estimar tamaño
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        dataset_count = len(lines)
        file_count = len(archivos_todos)
        
        # Estrategia 1: Buscar Metadata "source" (Futuro - 100% fiable)
        content_str = "".join(lines)
        if '"source":' in content_str:
            print("  [INFO] Metadata 'source' encontrada. Reconstruyendo con precisión...")
            for archivo in archivos_todos:
                if f'"source": "{archivo}"' in content_str:
                    processed.add(archivo)
        
        # Estrategia 2: Heurística por volumen (Fix para estado heredado)
        # Si hay muchas conversaciones (ej: > 3 por archivo) y el log está vacío, asumir todo procesado
        # para evitar duplicados masivos.
        elif dataset_count > (file_count * 2) and file_count > 0:
            print(f"  [INFO] Dataset grande ({dataset_count} convs) vs archivos ({file_count}).")
            print("  [INFO] Asumiendo que todos los archivos anteriores ya fueron procesados.")
            processed = set(archivos_todos)
            
        else:
            # Estrategia 3: Búsqueda difusa por nombre (Fallback débil)
            content_lower = content_str.lower()
            for archivo in archivos_todos:
                name_clean = archivo.replace('_refined.md', '').replace('_refined.txt', '')
                parts = name_clean.split('-')
                search_term = parts[0].strip().lower()
                if len(search_term) < 4:
                    search_term = name_clean.replace('_', ' ').lower()

                if search_term in content_lower:
                    processed.add(archivo)

        # Guardar resultado
        if processed:
            with open(log_path, 'w', encoding='utf-8') as f_log:
                for archivo in sorted(list(processed)):
                    f_log.write(archivo + "\n")
            print(f"  [INFO] Log reconstruido exitosamente con {len(processed)} archivos.")
        
        return processed

    except Exception as e:
        print(f"  [WARN] Falló la reconstrucción del log: {e}")
        return set()


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] No se encontró: {INPUT_DIR}")
        print("        Ejecuta primero: python src/ai_text_cleaner.py")
        sys.exit(1)

    os.makedirs(DATASETS_DIR, exist_ok=True)

    # Buscar archivos refinados
    archivos_todos = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.endswith('_refined.md') or f.endswith('_refined.txt')
    ])

    # Cargar registro de archivos ya procesados
    processed_files = set()
    
    if os.path.exists(ARCHIVO_LOG):
        try:
            with open(ARCHIVO_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_files.add(line.strip())
        except Exception as e:
            print(f"[WARN] No se pudo leer el archivo de log: {e}")
    elif os.path.exists(SALIDA_GENERICO) and os.path.getsize(SALIDA_GENERICO) > 0:
        # Caso: Existe dataset pero no log. Intentar reconstruir.
        processed_files = reconstruct_log_from_dataset(archivos_todos, ARCHIVO_LOG, SALIDA_GENERICO)
    
    # Filtrar pendientes
    archivos_pendientes = [f for f in archivos_todos if f not in processed_files]
    
    total_todos = len(archivos_todos)
    total_pendientes = len(archivos_pendientes)
    total_saltados = total_todos - total_pendientes

    print(f"Archivos totales    : {total_todos}")
    print(f"Ya procesados       : {total_saltados}")
    print(f"Pendientes          : {total_pendientes}")
    print(f"Entrada             : {INPUT_DIR}")
    print(f"Salida genérica     : {SALIDA_GENERICO}")
    print(f"Salida Gemini       : {SALIDA_GEMINI}")
    print("=" * 60)

    if total_pendientes == 0:
        print("[INFO] No hay archivos nuevos para procesar.")
        return

    total_convs = 0
    total_turnos = 0
    exitosos = 0
    fallidos = 0

    for i, archivo in enumerate(archivos_pendientes, 1):
        ruta = os.path.join(INPUT_DIR, archivo)
        print(f"\n[{i}/{total_pendientes}] {archivo}")

        conversaciones = procesar_archivo(ruta)

        if conversaciones:
            # Guardar ambos formatos
            with open(SALIDA_GENERICO, 'a', encoding='utf-8') as fg, \
                 open(SALIDA_GEMINI, 'a', encoding='utf-8') as fgem:
                for conv in conversaciones:
                    # Agregar metadata de origen para trazabilidad futura
                    conv_con_meta = conv.copy()
                    conv_con_meta["source"] = archivo
                    
                    # Formato genérico (messages)
                    fg.write(json.dumps(conv_con_meta, ensure_ascii=False) + "\n")
                    # Formato Gemini native (contents)
                    fgem.write(json.dumps(conv_a_gemini(conv), ensure_ascii=False) + "\n")
            
            # Actualizar log de procesados
            with open(ARCHIVO_LOG, 'a', encoding='utf-8') as f_log:
                f_log.write(archivo + "\n")

            turnos = sum(len(c["messages"]) for c in conversaciones)
            total_convs += len(conversaciones)
            total_turnos += turnos
            exitosos += 1

            # Vista previa: primer intercambio de la primera conversación
            preview = conversaciones[0]["messages"]
            print(f"    User : {preview[0]['content'][:80]}...")
            print(f"    Model: {preview[1]['content'][:80]}...")
        else:
            fallidos += 1

        if i < total_pendientes:
            time.sleep(PAUSA_ENTRE_ARCHIVOS)

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print(f"  Archivos nuevos procesados : {exitosos}")
    print(f"  Archivos fallidos          : {fallidos}")
    print(f"  Archivos previamente saltados: {total_saltados}")
    print(f"  Conversaciones generadas   : {total_convs}")
    print(f"  Turnos totales             : {total_turnos}")
    print(f"  Genérico                   : {SALIDA_GENERICO}")
    print(f"  Gemini                     : {SALIDA_GEMINI}")
    print("=" * 60)


if __name__ == "__main__":
    main()
