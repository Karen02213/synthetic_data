import json
import random
import time
import re
import os
from difflib import SequenceMatcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# ============================================================
# CONFIGURACIÓN
# ============================================================
MI_CLAVE_GOOGLE = "AIzaSyAsqHzC0o5z72JCALqMfYKgIGRlSapgEFg"
ARCHIVO_ENTRENAMIENTO = "dataset_veterinario_limpio.jsonl"
ARCHIVO_VALIDACION = "dataset_validacion_ia.jsonl"
TAMANO_BATCH = 20          # Procesar de 20 en 20
TOTAL_A_GENERAR = 80      # Total de registros de validación a generar
MAX_REINTENTOS = 3         # Reintentos por cada item si falla
UMBRAL_SIMILITUD = 0.55    # Máxima similitud permitida (0-1). Menor = más diferente
MIN_LONGITUD_PREGUNTA = 15 # Caracteres mínimos para una pregunta válida
MIN_LONGITUD_RESPUESTA = 30 # Caracteres mínimos para una respuesta válida

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    temperature=0.8,
    google_api_key=MI_CLAVE_GOOGLE
)

# ============================================================
# PROMPT ROBUSTO
# ============================================================
TEMPLATE_VALIDACION = """
Eres un veterinario experto y también un especialista en generación de datos sintéticos de alta calidad para fine-tuning de modelos de lenguaje.

Tu tarea es generar un par pregunta/respuesta para un conjunto de VALIDACIÓN.
Este par debe evaluar el MISMO conocimiento veterinario que el ejemplo original, pero debe ser lo suficientemente diferente para que no sea una simple paráfrasis.

--- EJEMPLO ORIGINAL (entrenamiento) ---
Pregunta: "{pregunta}"
Respuesta: "{respuesta}"
--- FIN DEL EJEMPLO ---

REGLAS ESTRICTAS:
1. La nueva pregunta DEBE cubrir el mismo tema veterinario, pero formulada de manera muy distinta:
   - Cambia la estructura gramatical (de pregunta directa a situación, de formal a coloquial, etc.)
   - Usa sinónimos y expresiones diferentes
   - Puedes plantearla como un escenario real de un dueño preocupado
   - NO copies frases textuales de la pregunta original
2. La nueva respuesta DEBE:
   - Ser factualmente correcta y coherente con la información original
   - Estar redactada con palabras y estructura DIFERENTE a la original
   - Ser clara, útil y completa (mínimo 2-3 oraciones)
   - Incluir al menos un dato práctico o consejo accionable para el dueño
   - NO inventar datos médicos que no estén implícitos en la respuesta original
3. Ambos textos deben estar en español natural y correcto.
4. NO incluyas saludos, despedidas ni frases como "¡Buena pregunta!".

Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta (sin texto adicional antes ni después):
{{
  "pregunta": "Tu nueva pregunta aquí",
  "respuesta": "Tu nueva respuesta aquí"
}}
"""

prompt_template = ChatPromptTemplate.from_template(TEMPLATE_VALIDACION)


# ============================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================
def calcular_similitud(texto_a, texto_b):
    """Calcula la similitud entre dos textos (0.0 = totalmente diferentes, 1.0 = idénticos)."""
    return SequenceMatcher(None, texto_a.lower(), texto_b.lower()).ratio()


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

    # Buscar el primer { ... } válido con regex
    match = re.search(r'\{[^{}]*"pregunta"\s*:\s*"[^"]+?"[^{}]*"respuesta"\s*:\s*"[^"]+?"[^{}]*\}', texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Último intento: buscar cualquier bloque JSON
    match = re.search(r'\{.*?\}', texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
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


# ============================================================
# FUNCIÓN PRINCIPAL CON BATCHES
# ============================================================
def generar_validacion_por_batches(archivo_entrada, archivo_salida, total, batch_size):
    """
    Genera datos de validación procesando en batches de tamaño definido.
    Hace append al archivo de salida en cada batch.
    """
    print("=" * 60)
    print("  GENERADOR DE VALIDACIÓN SINTÉTICA (por batches)")
    print("=" * 60)

    # 1. Leer datos de entrenamiento
    try:
        with open(archivo_entrada, 'r', encoding='utf-8') as f:
            datos_entrenamiento = []
            for num_linea, linea in enumerate(f, 1):
                if linea.strip():
                    try:
                        datos_entrenamiento.append(json.loads(linea))
                    except json.JSONDecodeError:
                        print(f"  ⚠ Línea {num_linea} con JSON inválido, saltando...")
    except FileNotFoundError:
        print(f"  ✘ Error: No se encontró '{archivo_entrada}'")
        return

    print(f"  Registros de entrenamiento cargados: {len(datos_entrenamiento)}")

    # 2. Cargar registros de validación ya existentes (para no duplicar)
    registros_validacion = cargar_registros_existentes(archivo_salida)
    ya_generados = len(registros_validacion)
    print(f"  Registros de validación existentes: {ya_generados}")

    # Ajustar cuántos faltan
    pendientes = total - ya_generados
    if pendientes <= 0:
        print(f"\n  ✔ Ya tienes {ya_generados} registros. No se necesitan más.")
        print(f"  (Si quieres más, aumenta TOTAL_A_GENERAR en la configuración)")
        return

    print(f"  Registros pendientes por generar: {pendientes}")

    # 3. Seleccionar muestra aleatoria (sin repetir las ya usadas)
    preguntas_ya_usadas = set()
    for reg in registros_validacion:
        try:
            preguntas_ya_usadas.add(reg["messages"][0]["content"].lower().strip())
        except (KeyError, IndexError):
            pass

    candidatos = []
    for item in datos_entrenamiento:
        try:
            pregunta = item["messages"][0]["content"].lower().strip()
            if pregunta not in preguntas_ya_usadas:
                candidatos.append(item)
        except (KeyError, IndexError):
            continue

    if len(candidatos) < pendientes:
        print(f"  ⚠ Solo hay {len(candidatos)} candidatos disponibles (no usados). Se usarán todos.")
        muestra = candidatos
    else:
        muestra = random.sample(candidatos, pendientes)

    # 4. Dividir en batches
    num_batches = (len(muestra) + batch_size - 1) // batch_size
    print(f"\n  Se procesarán {len(muestra)} items en {num_batches} batch(es) de hasta {batch_size}")
    print("-" * 60)

    # Contadores globales
    total_exitosos = 0
    total_fallidos = 0
    total_rechazados = 0

    for batch_num in range(num_batches):
        inicio = batch_num * batch_size
        fin = min(inicio + batch_size, len(muestra))
        batch_actual = muestra[inicio:fin]

        print(f"\n  ▶ BATCH {batch_num + 1}/{num_batches} ({len(batch_actual)} items)")
        print(f"    Items {inicio + 1} a {fin}")

        registros_batch = []

        for i, item in enumerate(batch_actual, 1):
            idx_global = inicio + i

            # Extraer pregunta y respuesta original
            msgs = item.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
            model_msg = next((m["content"] for m in msgs if m["role"] == "model"), None)

            if not user_msg or not model_msg:
                print(f"    [{idx_global}] ✘ Registro sin pregunta/respuesta válida, saltando.")
                total_fallidos += 1
                continue

            print(f"    [{idx_global}/{len(muestra)}] Reformulando: \"{user_msg[:50]}...\"")

            # Intentar con reintentos
            exito = False
            for intento in range(1, MAX_REINTENTOS + 1):
                try:
                    mensaje = prompt_template.format_messages(pregunta=user_msg, respuesta=model_msg)
                    respuesta_ia = chat_model.invoke(mensaje)

                    # Extraer JSON robusto
                    dato_nuevo = extraer_json_de_respuesta(respuesta_ia.content)
                    if dato_nuevo is None:
                        print(f"             Intento {intento}: JSON inválido, reintentando...")
                        time.sleep(1)
                        continue

                    # Validar calidad
                    es_valido, razon = validar_registro(dato_nuevo, user_msg, model_msg)
                    if not es_valido:
                        print(f"             Intento {intento}: Rechazado → {razon}")
                        total_rechazados += 1
                        time.sleep(0.5)
                        continue

                    # Construir registro final
                    registro_final = {
                        "messages": [
                            {"role": "user", "content": dato_nuevo["pregunta"].strip()},
                            {"role": "model", "content": dato_nuevo["respuesta"].strip()}
                        ]
                    }

                    # Verificar duplicado contra los ya generados
                    todos_existentes = registros_validacion + registros_batch
                    if verificar_duplicados(registro_final, todos_existentes):
                        print(f"             Intento {intento}: Duplicado detectado, reintentando...")
                        total_rechazados += 1
                        time.sleep(0.5)
                        continue

                    # ¡Todo bien!
                    registros_batch.append(registro_final)
                    sim_p = calcular_similitud(dato_nuevo["pregunta"], user_msg)
                    sim_r = calcular_similitud(dato_nuevo["respuesta"], model_msg)
                    print(f"             ✔ Generado (sim. pregunta: {sim_p:.0%}, resp: {sim_r:.0%})")
                    exito = True
                    break

                except Exception as e:
                    print(f"             Intento {intento}: Error → {e}")
                    time.sleep(2)

            if not exito:
                print(f"             ✘ Falló tras {MAX_REINTENTOS} intentos.")
                total_fallidos += 1

        # 5. Guardar batch en archivo (APPEND)
        if registros_batch:
            with open(archivo_salida, 'a', encoding='utf-8') as f_out:
                for reg in registros_batch:
                    f_out.write(json.dumps(reg, ensure_ascii=False) + "\n")

            total_exitosos += len(registros_batch)
            registros_validacion.extend(registros_batch)

            print(f"\n    ✔ Batch {batch_num + 1} guardado: {len(registros_batch)} registros → {archivo_salida}")
        else:
            print(f"\n    ⚠ Batch {batch_num + 1}: No se generaron registros válidos.")

        # Pausa entre batches para no saturar la API
        if batch_num < num_batches - 1:
            print(f"    ⏳ Pausa de 3 segundos antes del siguiente batch...")
            time.sleep(3)

    # 6. Resumen final
    total_final = len(cargar_registros_existentes(archivo_salida))
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    print(f"  ✔ Nuevos registros generados: {total_exitosos}")
    print(f"  ✘ Fallidos (sin resultado):   {total_fallidos}")
    print(f"  ⚠ Rechazados por validación:  {total_rechazados}")
    print(f"  📄 Total en archivo validación: {total_final}")
    print(f"  📁 Archivo: {archivo_salida}")

    if total_final >= 10:
        print(f"\n  ✔ Tienes {total_final} registros. ¡Listo para subir a Google como set de validación!")
    else:
        print(f"\n  ⚠ Tienes {total_final} registros. Google pide mínimo 10. Ejecuta de nuevo para generar más.")

    print("=" * 60)


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    generar_validacion_por_batches(
        archivo_entrada=ARCHIVO_ENTRENAMIENTO,
        archivo_salida=ARCHIVO_VALIDACION,
        total=TOTAL_A_GENERAR,
        batch_size=TAMANO_BATCH,
    )
