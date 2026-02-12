import os
import sys
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# CONFIGURACIÓN
# ============================================================
MI_CLAVE_GOOGLE = "AIzaSyAsqHzC0o5z72JCALqMfYKgIGRlSapgEFg"

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'cleaned_text')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'refined_text')

# Control de flujo
MAX_REINTENTOS = 3
PAUSA_ENTRE_LLAMADAS = 2
PAUSA_REINTENTO = 5
MIN_CHARS_ENTRADA = 50
MIN_CHARS_SALIDA = 30

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.15,
    google_api_key=MI_CLAVE_GOOGLE)

# ============================================================
# PROMPT - Uno por archivo (fiable y simple)
# ============================================================
TEMPLATE_LIMPIEZA = """Eres un editor profesional especializado en textos técnicos veterinarios.
Tu única tarea es RESTAURAR el siguiente texto extraído de un PDF mediante OCR.

El texto proviene de artículos del Manual Veterinario Merck (versión para dueños de mascotas).

=== REGLAS ESTRICTAS ===

1. ELIMINAR RUIDO (quitar completamente):
   - Encabezados y pies de página repetidos (ej: "MANUAL DE MERCK", "Manual veterinario")
   - URLs, enlaces web y rutas de navegación (ej: "nepsuoun merchvetmanualconi...")
   - Números de página sueltos
   - Menús de navegación del sitio web
   - Texto como "VERSION PARA DUEÑOS DE MASCOTAS"
   - Líneas de dimensiones o metadatos (ej: "arms, 628mm")
   - Fragmentos como "Ver también contenido profesional sobre..."
   - Cualquier artefacto visual que no sea parte del contenido educativo

2. CORREGIR ERRORES DE OCR:
   - Caracteres mal reconocidos: é por D, ó por 0, í por l, ñ por fi, etc.
   - Palabras cortadas o fusionadas incorrectamente
   - Acentos y tildes faltantes en español
   - Puntuación incorrecta o faltante
   - Ejemplos comunes: "tDxicas"->"tóxicas", "célldas"->"cálidas", "posiblidades"->"posibilidades"

3. FORMATO Y ESTRUCTURA:
   - Usa el título del artículo como encabezado principal (# Título)
   - Si hay secciones claras, usa subtítulos (## Subtítulo)
   - Separa en párrafos coherentes y bien definidos
   - Mantén listas si el original las tiene
   - NO uses indentación innecesaria

4. INTEGRIDAD DEL CONTENIDO:
   - NO resumas, NO acortes, NO omitas información
   - Si una palabra está parcialmente ilegible pero el contexto veterinario la hace obvia, restáurala
   - NO agregues información nueva que no esté implícita en el texto
   - Mantén nombres de autores, fechas de revisión y datos bibliográficos si aparecen
   - Conserva el idioma original del texto (español o inglés)

5. COMPLEMENTAR CON CUIDADO:
   - SOLO completa palabras cortadas o frases rotas donde el sentido es 100%% claro
   - Si una oración está incompleta y no puedes deducir el final con certeza, déjala como está
   - Marca con [?] cualquier parte donde no estés seguro de la corrección

=== TEXTO A PROCESAR ===

Devuelve SOLO el texto restaurado, sin explicaciones ni comentarios adicionales.

TEXTO A RESTAURAR:
{texto}
"""

prompt_single = ChatPromptTemplate.from_template(TEMPLATE_LIMPIEZA)


def obtener_texto_seguro(respuesta):
    """Extrae texto de la respuesta IA, manejando list o str."""
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


def procesar_archivo(nombre, contenido):
    """Procesa un solo archivo con reintentos. Devuelve texto limpio o None."""
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            mensaje = prompt_single.format_messages(texto=contenido)
            respuesta = chat_model.invoke(mensaje)
            texto_limpio = obtener_texto_seguro(respuesta).strip()

            if len(texto_limpio) < MIN_CHARS_SALIDA:
                print(f"    [WARN] Intento {intento}: respuesta muy corta ({len(texto_limpio)} chars)")
                time.sleep(PAUSA_REINTENTO)
                continue

            return texto_limpio

        except Exception as e:
            print(f"    [ERROR] Intento {intento}/{MAX_REINTENTOS}: {e}")
            if intento < MAX_REINTENTOS:
                time.sleep(PAUSA_REINTENTO)

    return None


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] No se encontró: {INPUT_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Escanear archivos pendientes
    todos = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('_cleaned.txt')])
    pendientes = []

    for archivo in todos:
        nombre_base = archivo.replace('_cleaned.txt', '')
        ruta_salida = os.path.join(OUTPUT_DIR, f"{nombre_base}_refined.md")
        if os.path.exists(ruta_salida):
            continue

        ruta_entrada = os.path.join(INPUT_DIR, archivo)
        try:
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                contenido = f.read()
            if len(contenido.strip()) >= MIN_CHARS_ENTRADA:
                pendientes.append((archivo, contenido))
        except Exception as e:
            print(f"  [SKIP] No se pudo leer {archivo}: {e}")

    total = len(pendientes)
    saltados = len(todos) - total

    if total == 0:
        print(f"No hay archivos pendientes ({saltados} ya procesados).")
        return

    print(f"Archivos totales : {len(todos)}")
    print(f"Ya procesados    : {saltados}")
    print(f"Pendientes       : {total}")
    print(f"Entrada          : {INPUT_DIR}")
    print(f"Salida           : {OUTPUT_DIR}")
    print("=" * 60)

    exitosos = 0
    fallidos = 0

    for i, (archivo, contenido) in enumerate(pendientes, 1):
        nombre_base = archivo.replace('_cleaned.txt', '')
        print(f"\n[{i}/{total}] {nombre_base} ({len(contenido)} chars)")

        resultado = procesar_archivo(archivo, contenido)

        if resultado:
            ruta_salida = os.path.join(OUTPUT_DIR, f"{nombre_base}_refined.md")
            with open(ruta_salida, 'w', encoding='utf-8') as f:
                f.write(resultado)
            print(f"  [OK] Guardado ({len(resultado)} chars)")
            exitosos += 1
        else:
            print(f"  [FAIL] No se pudo procesar después de {MAX_REINTENTOS} intentos")
            fallidos += 1

        if i < total:
            time.sleep(PAUSA_ENTRE_LLAMADAS)

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print(f"  Exitosos  : {exitosos}")
    print(f"  Fallidos  : {fallidos}")
    print(f"  Saltados  : {saltados}")
    print("=" * 60)


if __name__ == "__main__":
    main()
