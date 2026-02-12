# Synthetic Data Generation Pipeline for Veterinary LLM

This project implements an end-to-end pipeline to generate high-quality synthetic datasets for fine-tuning Large Language Models (LLMs) in the veterinary domain. It processes raw PDF documents, cleans them using AI, generates multi-turn conversations, and creates robust validation sets.

## 🚀 Features

- **OCR Processing**: Extracts text from PDFs (scanned or digital) using Tesseract and PyMuPDF.
- **AI Text Cleaning**: Uses Gemini to restore broken OCR text, correct typos, and structure content into Markdown.
- **Synthetic Conversation Generation**: Converts static text into realistic, multi-turn veterinary consultations (User/Vet Assistant).
- **Validation Set Generation**: Automatically creates a validation dataset ensuring no data leakage from training data.
- **Robustness**: Scripts support resuming work (skipping processed files) and appending to existing datasets.

## 📂 Project Structure

```
synthetic_data/
├── README.md               # This documentation
├── requirements.txt        # Python dependencies
├── data/
│   ├── pdfs/               # Input: Raw PDF files
│   ├── cleaned_text/       # Intermediate: OCR output (.txt) & CSVs
│   ├── refined_text/       # Intermediate: AI-cleaned Markdown files
│   ├── datasets/           # Output: Final JSONL datasets
│   │   ├── dataset_veterinario_limpio.jsonl  # Training data (Generic format)
│   │   ├── dataset_veterinario_gemini.jsonl  # Training data (Gemini format)
│   │   ├── dataset_validacion_ia.jsonl       # Validation data
│   │   └── processed_log.txt                 # Tracker for processed files (prevents duplicates)
│   ├── GenerarDataset_output.txt             # Execution log: Dataset generation results
│   ├── TextCleaner_output.txt                # Execution log: AI Cleaner results
│   └── totalTokens_cleaned_text_output.txt   # Report: Token usage stats per file
└── src/                    # Source code
    ├── ocr_processor.py         # Step 1: PDF to Text
    ├── ai_text_cleaner.py       # Step 2: Text Cleaning with LLM
    ├── generar_dataset.py       # Step 3: Text to Conversation
    ├── generar_validacion_ia.py # Step 4: Validation Set Generation
    ├── count_tokens.py          # Utility: Token stats
    └── list_models.py           # Utility: Check available Gemini models
```

## 🛠️ Prerequisites

- **Python 3.10+**
- **Tesseract OCR**: Must be installed on your system.
  - Linux: `sudo apt-get install tesseract-ocr poppler-utils`
  - Windows/Mac: Install via binary installer or brew.
- **Google AI Studio API Key**: Required for Gemini access.

## 📦 Installation

1. Clone the repository (if applicable) or navigate to the project folder.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure `requirements.txt` includes: `langchain-google-genai`, `langchain`, `pytesseract`, `pdf2image`, `pymupdf`, `pandas`, `langdetect`.*

3. **Configuration**: 
   The scripts currently use a hardcoded API key variable `MI_CLAVE_GOOGLE`. Ideally, set this up or ensure your environment supports it. Open `src/ai_text_cleaner.py` regarding configuration if needed.

## ⚙️ Usage Pipeline

Run the scripts in the following order to build your dataset.

### 1. Process PDFs (OCR)
Extracts raw text from PDFs located in `data/pdfs/`.
- **Input**: `data/pdfs/*.pdf`
- **Output**: `data/cleaned_text/*_cleaned.txt`
- Skips files already processed.

```bash
python src/ocr_processor.py
```

### 2. Clean Text (AI Restoration)
Uses an LLM to fix OCR errors, typos, and format text into clean Markdown.
- **Input**: `data/cleaned_text/*.txt`
- **Output**: `data/refined_text/*_refined.md`
- Skips files already processed.

```bash
python src/ai_text_cleaner.py
```

### 3. Generate Training Dataset
Converts refined text into multi-turn conversations between a pet owner and "GoodPawies" (AI Vet).
- **Input**: `data/refined_text/*.md`
- **Output**: Appends to `data/datasets/dataset_veterinario_limpio.jsonl`
- Tracks processed files in `processed_log.txt` to avoid duplicates.

```bash
python src/generar_dataset.py
```

### 4. Generate Validation Dataset
Creates a distinct validation set by reformulating training examples to ensure robust model evaluation.
- **Input**: `data/datasets/dataset_veterinario_limpio.jsonl`
- **Output**: Appends to `data/datasets/dataset_validacion_ia.jsonl`
- Uses similarity checks to prevent leakage or duplicates.

```bash
python src/generar_validacion_ia.py
```

## 📊 Output Formats

### JSONL (Generic)
Used for fine-tuning generic models or local LLMs.
```json
{
  "messages": [
    {"role": "user", "content": "My dog is scratching a lot."},
    {"role": "model", "content": "It could be allergies..."}
  ]
}
```

### JSONL (Gemini)
Native format for Google Vertex AI / Gemini Fine-tuning.
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "My dog is scratching a lot."}]},
    {"role": "model", "parts": [{"text": "It could be allergies..."}]}
  ]
}
```

## � Logs & Reports

The `data/` directory includes specific text files that help track the pipeline's progress and costs:

- **`totalTokens_cleaned_text_output.txt`**: Generated by `count_tokens.py`. Provides a detailed breakdown of token counts for each file (using `cl100k_base` encoding). This is crucial for estimating the costs of AI processing.
- **`GenerarDataset_output.txt`**: A log capture of the `generar_dataset.py` execution. It shows which files were processed, how many conversations were generated, and a summary of success/failure rates.
- **`TextCleaner_output.txt`**: A log capture of the `ai_text_cleaner.py` execution, detailing the cleaning process for each file.
- **`dataset_veterinario_limpio/gemini.jsonl`**: These are the final output files containing the synthetic training data.

## �📈 Utilities

- **Count Tokens**: Check how many tokens your cleaned text consumes.
  ```bash
  python src/count_tokens.py
  ```
- **List Models**: See available Google Gemini models.
  ```bash
  python src/list_models.py
  ```
