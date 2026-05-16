import os
import pypdf
import docx
from datetime import datetime
from pathlib import Path

# Helper functions for getting metadata and reading different file types:

def _get_file_metadata(filepath: str) -> dict:
    path = Path(filepath)
    
    file_metadata = {
        "filename": path.name,
        "file_type": path.suffix.replace('.', ''),
        "size_bytes": path.stat().st_size,
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    }

    return file_metadata

# txt parser
def read_txt(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# pdf parser
def read_pdf(filepath: str) -> str:
    reader = pypdf.PdfReader(filepath)

    if reader.is_encrypted:
        raise ValueError("File is encrypted. Cannot read content.")
    
    text_chunks = []
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text_chunks.append(extracted_text)        
    
    final_text = "\n".join(text_chunks) 

    if not final_text.strip():
        raise ValueError(f"Warning: No text found in '{filepath}'. It might be a scanned image.")
    
    return final_text

# docx parser
def read_docx(filepath: str) -> str:
    doc = docx.Document(filepath)
    
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    final_text = "\n".join(paragraphs)
    
    if not final_text.strip():
        raise ValueError(f"Warning: No text found in '{filepath}'. It might be an empty document.")
    
    return final_text

# Tool 1 - read_file(filepath: str) → dict
def read_file(filepath: str) -> dict:
    try:
        file_data = {}
        file_metadata = _get_file_metadata(filepath)
        file_data["metadata"] = file_metadata
        file_type = file_metadata["file_type"].lower()

        if file_type == "txt":
            content = read_txt(filepath)
        elif file_type == "pdf":
            content = read_pdf(filepath)  # Placeholder for PDF reading function
        elif file_type == "docx":
            content = read_docx(filepath)  # Placeholder for DOCX reading function
        else:
            raise ValueError("Unsupported file type. Supported types are: txt, pdf, docx") 
        
        file_data["content"] = content
        file_data["status"] = "success"

        return file_data
        
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"Could not find the file at path: {filepath}",
            "content": None
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"An error occurred while reading the file: {str(e)}",
            "content": None
        }

    

# Tool 2 - list_files(directory: str, extension: str = None) → list
def list_files(directory: str, extension: str = None) -> list:
    try:   
        path = Path(directory)
        
        if not path.is_dir():
            return [{"error": f"The provided path '{directory}' is not a valid directory."}]
        
        target_ext = None
        if extension:
            target_ext = f".{extension.lower().lstrip('.')}"
        
        files = []
        for entry in path.iterdir():
            if entry.is_file():
                if target_ext is None or entry.suffix.lower() == target_ext:
                    files.append(_get_file_metadata(str(entry)))
    
        return files
    
    except Exception as e:
        return [{"error": f"An error occurred while listing files: {str(e)}"}]



# Tool 3 - write_file(filepath: str, content: str) → dict
def write_file(filepath: str, content: str) -> dict:
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as file:
            file.write(content)
            
        return {
            "status": "success",
            "message": f"Successfully wrote content to {path.name}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to write file: {str(e)}"
        }

# Tool 4 - search_in_file(filepath: str, keyword: str) → dict
def search_in_file(filepath: str, keyword: str) -> dict:
    try:
        file_data = read_file(filepath)
        
        if file_data["status"] == "error":
            return file_data
            
        content = file_data.get("content", "")
        if not content:
            return {"status": "success", "matches": [], "message": "File is empty."}
            
        matches = []
        lines = content.split('\n')
        keyword_lower = keyword.lower()
        
        for line in lines:
            if keyword_lower in line.lower():
                matches.append(line.strip())
                
        return {
            "status": "success",
            "keyword_searched": keyword,
            "total_matches": len(matches),
            "matches": matches
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}"
        }
