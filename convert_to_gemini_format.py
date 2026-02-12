import json
import os

def convertir_formato_gemini(archivo_entrada, archivo_salida):
    """
    Convierte un dataset formato 'ChatCompletions' (OpenAI) a 'GenerateContent' (Gemini Native).
    
    ENTRADA (OpenAI style):
    {"messages": [{"role": "user", "content": "Hola"}, {"role": "model", "content": "Hola"}]}
    
    SALIDA (Gemini style):
    {"contents": [{"role": "user", "parts": [{"text": "Hola"}]}, {"role": "model", "parts": [{"text": "Hola"}]}]}
    """
    
    print(f"Procesando {archivo_entrada} -> {archivo_salida}")
    
    if not os.path.exists(archivo_entrada):
        print(f"Error: No se encuentra {archivo_entrada}")
        return

    registros_convertidos = 0
    errores = 0

    with open(archivo_entrada, 'r', encoding='utf-8') as fin, \
         open(archivo_salida, 'w', encoding='utf-8') as fout:
        
        for linea in fin:
            if not linea.strip():
                continue
                
            try:
                data = json.loads(linea)
                messages = data.get("messages", [])
                
                # Nueva estructura para Gemini
                gemini_entry = {
                    "contents": []
                }
                
                es_valido = True
                
                for msg in messages:
                    rol = msg.get("role")
                    contenido = msg.get("content")
                    
                    # Mapeo de roles si fuera necesario (assistant -> model)
                    if rol == "assistant":
                        rol = "model"
                    
                    if not rol or not contenido:
                        es_valido = False
                        break
                        
                    # Estructura native: role + parts list
                    part = {
                        "role": rol,
                        "parts": [{"text": contenido}]
                    }
                    gemini_entry["contents"].append(part)
                
                if es_valido and len(gemini_entry["contents"]) > 0:
                    fout.write(json.dumps(gemini_entry, ensure_ascii=False) + "\n")
                    registros_convertidos += 1
                else:
                    errores += 1
                    
            except json.JSONDecodeError:
                errores += 1
                print("Error de JSON en una línea")

    print(f"Completado: {registros_convertidos} registros convertidos. {errores} errores.\n")

if __name__ == "__main__":
    # 1. Convertir Dataset de Entrenamiento
    convertir_formato_gemini(
        "dataset_veterinario_limpio.jsonl", 
        "dataset_veterinario_gemini.jsonl"
    )
    
    # 2. Convertir Dataset de Validación
    convertir_formato_gemini(
        "dataset_validacion_ia.jsonl", 
        "dataset_validacion_gemini_native.jsonl"
    )
    
    print("¡Listo! Usa los archivos terminados en '_gemini.jsonl' para tu fine-tuning.")
