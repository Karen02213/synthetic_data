import os
import argparse
import sys

# Try to import tiktoken for accurate counting (standard for OpenAI, good approximation for others)
# If not available, use a fallback approximation
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

def count_tokens(text):
    """
    Count tokens using tiktoken (cl100k_base) or fallback to word-count approximation.
    """
    if HAS_TIKTOKEN:
        # cl100k_base is used by GPT-4 and is a common standard for token estimation
        encoder = tiktoken.get_encoding("cl100k_base")
        # encode returns list of integers
        return len(encoder.encode(text))
    else:
        # Fallback: Approximate tokens.
        # A common rule of thumb: 1000 tokens ~= 750 words (0.75 words/token)
        # So tokens = words / 0.75 = words * 1.33
        words = text.split()
        return int(len(words) * 1.33)

def get_output_path(folder_path):
    # Determine the project data directory to store outputs consistently
    # Assuming this script is in src/, project root is one level up
    # But flexibility is better.
    
    folder_name = os.path.basename(os.path.normpath(folder_path))
    filename = f"totalTokens_{folder_name}_output.txt"
    
    # Try to find the 'data' directory in the project root
    # If the input folder is inside 'data', write to 'data'.
    # Otherwise, write to current directory or handle gracefully.
    
    # Let's save it directly in the 'data' folder if it exists in the parent of src
    # Or parallel to the script?
    
    # Strategy: 
    # 1. If currently in project root and data exists, stick it there.
    # 2. Else, stick it in the parent directory of the scanned folder?
    # User's other output files are in `data/`.
    
    base_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(base_dir)              # project/
    data_dir = os.path.join(project_root, 'data')
    
    if os.path.isdir(data_dir):
        return os.path.join(data_dir, filename)
    
    return filename # Fallback to current dir

def main():
    parser = argparse.ArgumentParser(description="Count tokens in .txt files within a specified folder.")
    
    # Default to data/cleaned_text relative to the project root assuming script is run from root
    # But let's make the default path absolute relative to this script for robustness
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cleaned_text')
    
    parser.add_argument('folder', nargs='?', default=default_path, 
                        help='Path to the folder containing .txt files (default: data/cleaned_text)')
    
    args = parser.parse_args()
    folder_path = args.folder

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    print(f"Scanning folder: {folder_path}...")
    if not HAS_TIKTOKEN:
        print("Warning: 'tiktoken' library not found. Process will use an approximation (words * ~1.33).")
        print("For accurate counts, install tiktoken: pip install tiktoken")

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    files.sort()
    
    results = []
    total_tokens_all = 0
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            t_count = count_tokens(content)
            results.append((filename, t_count))
            total_tokens_all += t_count
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Write output
    output_file = get_output_path(folder_path)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Token Count Report\n")
            f.write(f"Folder: {folder_path}\n")
            f.write(f"Method: {'tiktoken (cl100k_base)' if HAS_TIKTOKEN else 'Approximation (1.33 * words)'}\n")
            f.write(f"Total Files: {len(results)}\n")
            f.write(f"TOTAL TOKENS: {total_tokens_all:,}\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Filename':<50} | {'Tokens':>10}\n")
            f.write("-" * 60 + "\n")
            
            for name, count in results:
                # Truncate long filenames for display
                display_name = (name[:47] + '..') if len(name) > 49 else name
                f.write(f"{display_name:<50} | {count:>10,}\n")
    
        print(f"\nSuccess! processed {len(results)} files.")
        print(f"Total Tokens: {total_tokens_all:,}")
        print(f"Report saved to: {output_file}")
        
    except IOError as e:
        print(f"Error writing report to {output_file}: {e}")

if __name__ == "__main__":
    main()
