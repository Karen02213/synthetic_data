import json
import random
import time
import re
import os
import sys
from difflib import SequenceMatcher
# Make sure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import count_tokens as ct

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# CONFIGURACIÓN
# ============================================================
MI_CLAVE_GOOGLE = "AIzaSyAsqHzC0o5z72JCALqMfYKgIGRlSapgEFg"

# Calculate paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')

ARCHIVO_ENTRENAMIENTO = os.path.join(DATASETS_DIR, "dataset_veterinario_limpio.jsonl")
ARCHIVO_VALIDACION = os.path.join(DATASETS_DIR, "dataset_validacion_ia.jsonl")
ARCHIVO_VALIDACION_GEMINI = os.path.join(DATASETS_DIR, "dataset_validacion_gemini_native.jsonl")
ARCHIVO_LOG = os.path.join(DATASETS_DIR, "validacion_processed_log.txt")

# Ajustes de Batch y Tokens
TOTAL_A_GENERAR = 260       # Objetivo total
MAX_INPUT_TOKENS = 300000  # Límite estricto de tokens de entrada
MAX_OUTPUT_ITEMS = 50      # Límite por batch para evitar cortar la respuesta (output token limit)

MAX_REINTENTOS = 3         # Reintentos por batch
UMBRAL_SIMILITUD = 0.55    # Máxima similitud permitida (0-1). Menor = más diferente
MIN_LONGITUD_PREGUNTA = 15 # Caracteres mínimos para una pregunta válida
MIN_LONGITUD_RESPUESTA = 30 # Caracteres mínimos para una respuesta válida

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", # Usamos modelo con buena ventana de contexto
    temperature=0.8,
    google_api_key=MI_CLAVE_GOOGLE
)

# ============================================================
# PROMPT BULK
# ============================================================
TEMPLATE_BULK = """
Eres un veterinario experto y especialista en generación de datos sintéticos de alta calidad para fine-tuning de modelos de lenguaje.

Tu tarea es generar pares de pregunta/respuesta de VALIDACIÓN a partir de una lista de ejemplos de entrenamiento.
Recibirás una lista JSON. Para CADA objeto en la lista, genera una variación que evalúe el MISMO conocimiento pero formulado de manera diferente.

REGLAS PARA CADA ITEM:
1. La nueva pregunta DEBE cubrir el mismo tema veterinario, pero formulada MUY distinta (sinónimos, cambio de estructura, caso clínico, coloquial, etc.).
2. La nueva respuesta DEBE ser factualmente correcta, coherente, útil (mínimo 2-3 oraciones), y NO copiar frases textuales de la original.
3. NO incluyas saludos, despedidas ni frases como "¡Buena pregunta!".
4. Ambos textos deben estar en español natural y correcto.
5. Conserva el campo "id" exacto de cada item de entrada.

ENTRADA:
{items_json}

SALIDA (devuelve ÚNICAMENTE un JSON array crudo, sin bloques markdown):
[
  {{
    "id": <mismo id del item>,
    "pregunta": "Tu nueva pregunta aquí",
    "respuesta": "Tu nueva respuesta aquí"
  }},
  ...
]
"""

prompt_template = ChatPromptTemplate.from_template(TEMPLATE_BULK)


# ============================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================
def calcular_similitud(texto_a, texto_b):
    """Calcula la similitud entre dos textos (0.0 = totalmente diferentes, 1.0 = idénticos)."""
    return SequenceMatcher(None, texto_a.lower(), texto_b.lower()).ratio()


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


def extraer_json_de_respuesta(texto_crudo):
    """
    Extrae un objeto JSON válido de la respuesta de la IA,
    manejando casos con markdown, texto extra, etc.
    """
    # Quitar bloques de código markdown
    texto = texto_crudo.replace("```json", "").replace("```", "").strip()

    # Intentar parsear directamente
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # Si es una lista
    match_list = re.search(r'\[\s*\{.*\}\s*\]', texto, re.DOTALL)
    if match_list:
        try:
            return json.loads(match_list.group())
        except json.JSONDecodeError:
            pass

    # Si es un objeto único (fallback para batches de 1)
    match = re.search(r'\{[^{}]*"pregunta"\s*:\s*"[^"]+?"[^{}]*"respuesta"\s*:\s*"[^"]+?"[^{}]*\}', texto, re.DOTALL)
    if match:
        try:
            return [json.loads(match.group())]
        except json.JSONDecodeError:
            pass

    return None


def validar_registro(dato_nuevo, pregunta_original, respuesta_original):
    """
    Valida que el registro generado cumpla con los estándares de calidad.
    Retorna (es_valido: bool, razon: str).
    """
    # Verificar que tenga las claves necesarias
    if "pregunta" not in dato_nuevo or "respuesta" not in dato_nuevo:
        return False, "Faltan claves 'pregunta' o 'respuesta'"

    nueva_pregunta = dato_nuevo["pregunta"].strip()
    nueva_respuesta = dato_nuevo["respuesta"].strip()

    # Verificar longitud mínima
    if len(nueva_pregunta) < MIN_LONGITUD_PREGUNTA:
        return False, f"Pregunta muy corta ({len(nueva_pregunta)} chars, mínimo {MIN_LONGITUD_PREGUNTA})"

    if len(nueva_respuesta) < MIN_LONGITUD_RESPUESTA:
        return False, f"Respuesta muy corta ({len(nueva_respuesta)} chars, mínimo {MIN_LONGITUD_RESPUESTA})"

    # Verificar que no sea copia exacta
    if nueva_pregunta.lower() == pregunta_original.lower():
        return False, "Pregunta idéntica a la original"

    if nueva_respuesta.lower() == respuesta_original.lower():
        return False, "Respuesta idéntica a la original"

    # Verificar similitud (debe ser suficientemente diferente)
    sim_pregunta = calcular_similitud(nueva_pregunta, pregunta_original)
    if sim_pregunta > UMBRAL_SIMILITUD:
        return False, f"Pregunta muy similar a la original (similitud: {sim_pregunta:.0%}, máx: {UMBRAL_SIMILITUD:.0%})"

    sim_respuesta = calcular_similitud(nueva_respuesta, respuesta_original)
    if sim_respuesta > UMBRAL_SIMILITUD:
        return False, f"Respuesta muy similar a la original (similitud: {sim_respuesta:.0%}, máx: {UMBRAL_SIMILITUD:.0%})"

    # Verificar que no tenga frases genéricas no deseadas
    frases_prohibidas = ["buena pregunta", "excelente pregunta", "hola", "¡claro!", "con gusto"]
    for frase in frases_prohibidas:
        if frase in nueva_respuesta.lower():
            return False, f"Respuesta contiene frase no deseada: '{frase}'"

    return True, "OK"


def conv_a_gemini(conv):
    """Convierte formato genérico a formato Gemini native."""
    contents = []
    for msg in conv["messages"]:
        contents.append({
            "role": msg["role"],
            "parts": [{"text": msg["content"]}]
        })
    return {"contents": contents}


def verificar_duplicados(registro_nuevo, registros_existentes):
    """Verifica que el nuevo registro no sea duplicado de uno ya generado."""
    nueva_pregunta = registro_nuevo["messages"][0]["content"]
    for existente in registros_existentes:
        pregunta_existente = existente["messages"][0]["content"]
        if calcular_similitud(nueva_pregunta, pregunta_existente) > 0.70:
            return True  # Es duplicado
    return False


def cargar_registros_existentes(archivo):
    """Carga registros existentes del archivo de validación (si existe)."""
    if not os.path.exists(archivo):
        return []
    registros = []
    with open(archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            if linea.strip():
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return registros


def cargar_log_procesados(log_path):
    """Carga el set de índices de entrenamiento ya procesados desde el log."""
    procesados = set()
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    try:
                        procesados.add(int(linea))
                    except ValueError:
                        pass
    return procesados


def guardar_indices_en_log(log_path, indices):
    """Agrega índices procesados al log (append)."""
    with open(log_path, 'a', encoding='utf-8') as f:
        for idx in indices:
            f.write(f"{idx}\n")


def procesar_batch(batch_items, indices_globales):
    """
    Envía un batch completo a la IA y procesa los resultados.
    Retorna lista de registros validados.
    """
    input_list = []
    # Mapa para validación posterior (id -> original data)
    mapa_originales = {}

    for idx, item in zip(indices_globales, batch_items):
        msgs = item.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        model_msg = next((m["content"] for m in msgs if m["role"] == "model"), "")
        
        if not user_msg: continue

        obj = {
            "id": idx,
            "pregunta_original": user_msg,
            "respuesta_original": model_msg
        }
        input_list.append(obj)
        mapa_originales[idx] = (user_msg, model_msg)

    if not input_list:
        return []

    items_json = json.dumps(input_list, ensure_ascii=False, indent=2)
    
    # Check tokens using imported module
    prompt_tokens = ct.count_tokens(TEMPLATE_BULK) + ct.count_tokens(items_json)
    print(f"    tokens estimados: {prompt_tokens} (límite: {MAX_INPUT_TOKENS})")

    if prompt_tokens > MAX_INPUT_TOKENS:
        print("    [WARN] Batch excede límite de tokens. Reduciendo a la mitad recursivamente...")
        mid = len(batch_items) // 2
        return procesar_batch(batch_items[:mid], indices_globales[:mid]) + \
               procesar_batch(batch_items[mid:], indices_globales[mid:])

    # Invocar IA
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            mensaje = prompt_template.format_messages(items_json=items_json)
            respuesta_ia = chat_model.invoke(mensaje)
            contenido = obtener_texto_seguro(respuesta_ia)

            # Debug: mostrar primeros chars de la respuesta
            preview = contenido[:200].replace('\n', ' ')
            print(f"      Intento {intento}: respuesta recibida ({len(contenido)} chars) → {preview}...")

            datos_nuevos = extraer_json_de_respuesta(contenido)

            if datos_nuevos is None:
                print(f"      Intento {intento}: No se pudo extraer JSON de la respuesta.")
                time.sleep(2)
                continue

            # Si vino un dict suelto, envolver en lista
            if isinstance(datos_nuevos, dict):
                datos_nuevos = [datos_nuevos]

            if not isinstance(datos_nuevos, list):
                print(f"      Intento {intento}: Respuesta no es una lista válida (tipo: {type(datos_nuevos).__name__}).")
                time.sleep(2)
                continue

            # Procesar y validar resultados
            resultados_validos = []
            rechazados = 0
            sin_id = 0

            for dato in datos_nuevos:
                if not isinstance(dato, dict):
                    continue

                idx = dato.get("id")
                if idx not in mapa_originales:
                    sin_id += 1
                    continue

                # Verificar que tenga las claves necesarias
                if "pregunta" not in dato or "respuesta" not in dato:
                    rechazados += 1
                    continue

                orig_q, orig_a = mapa_originales[idx]
                es_valido, razon = validar_registro(dato, orig_q, orig_a)

                if es_valido:
                    registro_final = {
                        "messages": [
                            {"role": "user", "content": dato["pregunta"].strip()},
                            {"role": "model", "content": dato["respuesta"].strip()}
                        ]
                    }
                    resultados_validos.append(registro_final)
                else:
                    rechazados += 1

            print(f"      → Válidos: {len(resultados_validos)}, Rechazados: {rechazados}, Sin ID match: {sin_id}")

            if resultados_validos:
                return resultados_validos
            else:
                print(f"      Intento {intento}: 0 válidos, reintentando...")
                time.sleep(2)
                continue

        except Exception as e:
            print(f"      Intento {intento} error: {type(e).__name__}: {e}")
            time.sleep(3)

    return []


# ============================================================
# FUNCIÓN PRINCIPAL BULK
# ============================================================
def generar_validacion_bulk(archivo_entrada, archivo_salida, archivo_salida_gemini, total_objetivo):
    """
    Genera datos de validación maximizando el uso de tokens por llamada.
    Genera ambos formatos: genérico y Gemini native.
    """
    print("=" * 60)
    print("  GENERADOR DE VALIDACIÓN SINTÉTICA (BULK / HIGH-THROUGHPUT)")
    print("=" * 60)

    # 1. Leer datos de entrenamiento
    datos_entrenamiento = []
    if os.path.exists(archivo_entrada):
        with open(archivo_entrada, 'r', encoding='utf-8') as f:
            for linea in f:
                if linea.strip():
                    try:
                        datos_entrenamiento.append(json.loads(linea))
                    except: pass
    
    print(f"  Entrenamiento total : {len(datos_entrenamiento)}")

    # 2. Leer validación existente
    datos_existentes = cargar_registros_existentes(archivo_salida)
    ya_generados = len(datos_existentes)
    print(f"  Validación existente: {ya_generados}")

    pendientes = total_objetivo - ya_generados
    if pendientes <= 0:
        print(f"  ✔ Meta cumplida ({total_objetivo}).")
        return

    print(f"  Objetivo a generar  : {pendientes}")

    # 3. Cargar log de índices ya procesados (o reconstruir si falta)
    indices_procesados = cargar_log_procesados(ARCHIVO_LOG)

    if not indices_procesados and ya_generados > 0:
        # Log missing but output exists — bootstrap log from existing outputs
        # Since validation items are reformulations, we can't match exactly.
        # Conservatively mark the first N training indices as used (N = ya_generados)
        # to avoid feeding the same source data again.
        print(f"  [INFO] Log vacío pero hay {ya_generados} registros existentes. Reconstruyendo log...")
        indices_procesados = set(range(ya_generados))
        guardar_indices_en_log(ARCHIVO_LOG, sorted(indices_procesados))
        print(f"  [INFO] Log reconstruido con {len(indices_procesados)} índices.")

    print(f"  Índices en log      : {len(indices_procesados)}")

    # 4. Filtrar candidatos (no procesados previamente según log)
    candidatos = []
    indices = []
    
    for i, item in enumerate(datos_entrenamiento):
        if i in indices_procesados:
            continue
        try:
            q = item["messages"][0]["content"]
            if len(q) > 0:
                candidatos.append(item)
                indices.append(i)
        except:
            pass

    if not candidatos:
        print("  [WARN] No quedan candidatos nuevos en el set de entrenamiento.")
        return

    # 5. Limitar a los necesarios (con un margen extra por fallos de validación)
    necesarios = int(pendientes * 1.5) # Pedimos 50% extra para cubrir rechazos
    if necesarios > len(candidatos):
        seleccion = candidatos
        seleccion_idx = indices
    else:
        # Selección aleatoria zip
        zipped = list(zip(candidatos, indices))
        sample = random.sample(zipped, necesarios)
        seleccion, seleccion_idx = zip(*sample)

    print(f"  Candidatos seleccionados para procesar: {len(seleccion)}")

    # 6. Procesar en Batches Grandes (limitados por MAX_OUTPUT_ITEMS para seguridad de respuesta)
    # Aunque tengamos 300k tokens de entrada, la salida suele estar limitada.
    # Por eso procesamos en bloques de ~50 items, que es seguro.
    
    batch_size = MAX_OUTPUT_ITEMS
    total_chunks = (len(seleccion) + batch_size - 1) // batch_size
    
    exitosos_sesion = 0

    for i in range(0, len(seleccion), batch_size):
        chunk_items = seleccion[i : i + batch_size]
        chunk_indices = seleccion_idx[i : i + batch_size]
        
        batch_num = (i // batch_size) + 1
        print(f"\n  ▶ Batch {batch_num}/{total_chunks} ({len(chunk_items)} items)...")
        
        nuevos = procesar_batch(list(chunk_items), list(chunk_indices))
        
        if nuevos:
            # Guardar ambos formatos (append)
            with open(archivo_salida, 'a', encoding='utf-8') as fg, \
                 open(archivo_salida_gemini, 'a', encoding='utf-8') as fgem:
                for n in nuevos:
                    fg.write(json.dumps(n, ensure_ascii=False) + "\n")
                    fgem.write(json.dumps(conv_a_gemini(n), ensure_ascii=False) + "\n")
            
            # Registrar índices procesados en el log
            guardar_indices_en_log(ARCHIVO_LOG, chunk_indices)
            
            count = len(nuevos)
            exitosos_sesion += count
            print(f"    ✔ Guardados {count} registros nuevos. Log actualizado.")
        else:
            print("    [FAIL] Batch sin resultados válidos.")

        # Verificar si ya cumplimos la meta
        if (ya_generados + exitosos_sesion) >= total_objetivo:
            print("\n  ✔ ¡Objetivo alcanzado durante el procesamiento!")
            break
            
        time.sleep(2)

    print("\n" + "=" * 60)
    print(f"  RESUMEN: Generados {exitosos_sesion} nuevos registros.")
    total_final = len(cargar_registros_existentes(archivo_salida))
    print(f"  Total Validacion   : {total_final}")
    print(f"  Genérico           : {archivo_salida}")
    print(f"  Gemini native      : {archivo_salida_gemini}")
    print("=" * 60)

if __name__ == "__main__":
    generar_validacion_bulk(ARCHIVO_ENTRENAMIENTO, ARCHIVO_VALIDACION, ARCHIVO_VALIDACION_GEMINI, TOTAL_A_GENERAR)
