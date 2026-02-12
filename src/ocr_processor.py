import os
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from langdetect import detect, LangDetectException
import pandas as pd
from PIL import Image
import io

def extract_images_from_pdf(pdf_path):
    """
    Convert PDF pages to images.
    Tries pdf2image first, falls back to PyMuPDF if poppler is missing.
    """
    images = []
    try:
        # Try using pdf2image (requires poppler installed system-wide)
        print("Attempting to convert PDF to images using pdf2image...")
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"pdf2image failed ({e}). Falling back to PyMuPDF...")
        # Fallback to PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)
    return images

def extract_text_from_images(images):
    """
    Uses pytesseract to extract text from a list of PIL images.
    """
    full_text = ""
    print(f"Extracting text from {len(images)} pages using OCR...")
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    return full_text

def clean_and_structure_text(raw_text):
    """
    Separates text into paragraphs based on empty lines.
    Cleans structural whitespace.
    """
    lines = raw_text.split('\n')
    paragraphs = []
    current_paragraph = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # Empty line indicates end of paragraph
            if current_paragraph:
                # Join the lines of the paragraph
                para_text = " ".join(current_paragraph)
                paragraphs.append(para_text)
                current_paragraph = []
        else:
            current_paragraph.append(stripped_line)
    
    # Add the last paragraph if exists
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

def detect_language_safe(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def process_pdf(pdf_path, output_dir):
    """
    Integrates the pipeline: PDF -> Images -> OCR Text -> Paragraphs -> Save
    """
    filename = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Skip if already processed
    txt_path = os.path.join(output_dir, f"{name_without_ext}_cleaned.txt")
    if os.path.exists(txt_path):
        return "skipped"
    
    print(f"Processing: {filename}")
    
    # 1. Extract Images
    images = extract_images_from_pdf(pdf_path)
    if not images:
        print(f"Could not extract images from {filename}")
        return

    # 2. OCR Extraction
    raw_text = extract_text_from_images(images)

    # 3. Cleaning and Structuring
    paragraphs = clean_and_structure_text(raw_text)

    # 4. Create DataFrame for Analysis/Saving
    data = []
    for i, para in enumerate(paragraphs):
        lang = detect_language_safe(para)
        data.append({
            "source_file": filename,
            "paragraph_id": i + 1,
            "text": para,
            "language": lang,
            "length": len(para)
        })
    
    df = pd.DataFrame(data)

    # 5. Save to File (CSV and TXT)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save TXT (Cleaned full text)
    txt_path = os.path.join(output_dir, f"{name_without_ext}_cleaned.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(paragraphs))
    print(f"Saved text to: {txt_path}")

    # Save CSV (Structured data)
    csv_path = os.path.join(output_dir, f"{name_without_ext}_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved CSV to: {csv_path}")
    return "processed"

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PDF_DIR = os.path.join(BASE_DIR, 'data', 'pdfs')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'cleaned_text')
    
    # Process all PDFs in the folder
    if os.path.exists(PDF_DIR):
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {PDF_DIR}")
        
        processed = 0
        skipped = 0
        
        for pdf_file in sorted(pdf_files):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            result = process_pdf(pdf_path, OUTPUT_DIR)
            if result == "skipped":
                skipped += 1
            else:
                processed += 1
                print("-" * 50)
        
        print(f"\nDone. Processed: {processed}, Skipped: {skipped} (already exist)")
    else:
        print(f"PDF Directory not found: {PDF_DIR}")
