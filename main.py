import os
import pytesseract
import json
from pdf2image import convert_from_path
import pdfplumber
from fastmcp import FastMCP
import uuid 
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import yaml
import re
load_dotenv()


DATA_DIR = "data"
TEMPLATES_DIR = "templates"

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
client=genai.Client()
mcp = FastMCP(name="PrivatePDF")

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def is_text_based(text, min_chars=50):
    return len(text.strip()) > min_chars

def ensure_templates_dir():
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)

def extract_with_pdfpl(path : str) :
    pages_data=[]
    with pdfplumber.open(path) as pd:
        for i , page in enumerate(pd.pages, start=1):
            text=page.extract_text() or ""
            tables=page.extract_tables() or []
            pages_data.append({
                "page_no" : i,
                "text" : text, 
                "tables" : tables,
                "source" : "pdfplumber"

            })
    return pages_data


def extract_with_OCR(path : str) :
    pages_data=[]
    images=convert_from_path(path)
    for i , page in enumerate(images, start=1):
        text=pytesseract.image_to_string(page)

        pages_data.append({
            "page_no" : i , 
            "text" : text,
            "tables" : [],
            "source" : "ocr"
        }
        )
    return pages_data

def find_existing_pdf(path):
    """Check if this PDF path was already ingested."""
    if not os.path.exists(DATA_DIR):
        return None
    for pdf_id in os.listdir(DATA_DIR):
        content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
        if os.path.isfile(content_path):
            with open(content_path) as f:
                data = json.load(f)
            if data["original_path"] == path:
                return pdf_id  
    return None  

def save_chunks_to_faiss(pdf_id, pages_data):
    """
    1. Cut text into chunks
    2. Turn chunks into vectors (numbers)
    3. Save to FAISS file
    """
    
    all_text = " ".join(p["text"] for p in pages_data)
    
    
    chunk_size = 500
    chunks = []
    for i in range(0, len(all_text), chunk_size):
        chunk = all_text[i:i+chunk_size]
        chunks.append(chunk)
    

    vectors = EMBED_MODEL.encode(chunks)
    
    # Step 4: Save to FAISS
    index = faiss.IndexFlatL2(384)  
    index.add(vectors)              
    
    faiss.write_index(index, f"data/{pdf_id}/index.faiss")
    
    
    with open(f"data/{pdf_id}/chunks.json", "w") as f:
        json.dump(chunks, f)
    
    return len(chunks)

def search_faiss(pdf_id, question):
    """Find chunks similar to question."""
    index_path = f"{DATA_DIR}/{pdf_id}/index.faiss"
    chunks_path = f"{DATA_DIR}/{pdf_id}/chunks.json"
    
    if not os.path.exists(index_path):
        return []
    
    
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    
    q_vector = EMBED_MODEL.encode([question])
    
    
    distances, indices = index.search(np.array(q_vector), 3)
    
    
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])
    
    return results

@mcp.tool

def ingest_pdf(path : str ):
    ensure_data_dir()

    pages_data=extract_with_pdfpl(path)
    full_text="".join(p["text"] for p in pages_data)

    if is_text_based(full_text):
        extraction_method="Pdf plumbber"
    else:
        extraction_method="ocr"
        pages_data=extract_with_OCR(path)
        full_text = "".join(p["text"] for p in pages_data)

    pdf_id=str(uuid.uuid4())[:8]
    pdffolder=os.path.join(DATA_DIR, pdf_id)
    os.makedirs(pdffolder)

    total_tables = sum(len(p["tables"]) for p in pages_data)

    record = {
        "pdf_id": pdf_id,
        "original_path": path,
        "extraction_method": extraction_method,
        "page_count": len(pages_data),
        "total_tables": total_tables,
        "pages": pages_data
    }
    
    
    content_path = os.path.join(pdffolder, "content.json")
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    
    semantic_chunks = save_chunks_to_faiss(pdf_id, pages_data)
    
    return {
        "pdf_id": pdf_id,
        "page_count": len(pages_data),
        "tables_found": total_tables,
        "method": extraction_method,
        "semantic_chunks" : semantic_chunks,
        "sample_text": full_text[:200] + "..." if len(full_text) > 200 else full_text
    }

@mcp.tool
def list_pdfs():
    results = []
    if not os.path.exists(DATA_DIR):
        return results
    for pdf_id in os.listdir(DATA_DIR):
        content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
        if os.path.isfile(content_path):
            with open(content_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append({
                    "pdf_id": data["pdf_id"],
                    "original_path": data["original_path"],
                    "page_count": data["page_count"],
                    "total_tables": data["total_tables"],
                    "method": data["extraction_method"]
                })
    return results

@mcp.tool
def query_pdf(pdf_id: str, question: str):
    chunks = search_faiss(pdf_id, question)
    
    if not chunks:
        return "No results found."
    
    return "Here are the most relevant sections:\n\n" + "\n\n---\n\n".join(chunks)

@mcp.tool
def extract_table(pdf_id: str, page_num: int, table_index: int = 0):
   
    content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
    
    with open(content_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    target_page = None
    for page in data["pages"]:
        if page["page_no"] == page_num:
            target_page = page  
            break
    
    
    if target_page is None:
        return {"error": f"Page {page_num} not found"}
    
    
    
    
    tables = target_page.get("tables", [])
    
    
    if not tables:
        return {"error": "No tables on this page"}
    
    
    if table_index >= len(tables):
        return {"error": f"Only {len(tables)} tables exist (you asked for #{table_index})"}
    
    
    raw_table = tables[table_index]
    
    
    clean_table = []
    for row in raw_table:
        clean_row = []
        for cell in row:
            if cell is None:
                cell = ""
            
            cell = str(cell).strip().replace("\n", " ")
            clean_row.append(cell)
        clean_table.append(clean_row)
    
    
    
   
    headers = clean_table[0]
    
    
    data_rows = clean_table[1:]
    
    
    result = []
    for row in data_rows:
        row_dict = {}
        for i, header in enumerate(headers):
            
            if i < len(row):
                row_dict[header] = row[i]
            else:
                row_dict[header] = ""
        result.append(row_dict)
    
    return {
        "page": page_num,
        "table_index": table_index,
        "headers": headers,
        "rows": len(result),
        "data": result
    }
@mcp.tool
def generate_template(pdf_id: str, name: str):
    """
    Uses Gemini AI to analyze PDF structure ONCE, saves rules as YAML.
    Next time: zero AI cost, instant extraction.
    """
    ensure_templates_dir()
    
    
    content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
    with open(content_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    
    pdf_summary = {
        "filename": data["original_path"],
        "pages": data["page_count"],
        "method": data["extraction_method"],
        "tables_per_page": [
            {"page": p["page_no"], "table_count": len(p["tables"])}
            for p in data["pages"]
        ],
        "sample_text_first_page": data["pages"][0]["text"][:200] if data["pages"] else "",
        "table_headers": [
        p["tables"][0][0]  
        for p in data["pages"] if p.get("tables")
    ]
        
    }
    
    
    prompt = f"""
You are a PDF structure analyzer. Your job is to write a YAML template that captures how to extract data from this type of PDF.

Here is the extracted data from a sample PDF:
{json.dumps(pdf_summary, indent=2)}

Write a YAML template with these sections:
1. `name`: the template name ({name})
2. `signature`: how to recognize this PDF type (unique text on first page, keywords)
3. `extraction`: what to extract and where (tables by page, metadata fields)

Rules:
- Use specific text snippets for signatures, not generic descriptions
- If tables exist, note exact page numbers and table indices
- Include `expected_columns` if you can infer them from table headers
- Keep it concrete enough that a Python script can follow it blindly

Here's an example of what I am expecting 
The extraction section must follow EXACTLY this structure:

extraction:
  rule_name:
    type: table
    page: 1
    table_index: 0
    expected_columns:
      - "Column1"
      - "Column2"
    
  another_rule:
    type: metadata
    page: 1
    fields:
      field_name:
        pattern: "regex here"

No nested lists. No extra nesting. Each key under extraction must have a 'type' field directly.
Return ONLY the YAML content. No markdown fences, no explanations.
"""
    
    
    response = client.models.generate_content(model="models/gemini-2.5-flash", contents=prompt)
    
    yaml_content = response.text.strip()
    
    
    if yaml_content.startswith("```yaml"):
        yaml_content = yaml_content[7:]
    if yaml_content.endswith("```"):
        yaml_content = yaml_content[:-3]
    yaml_content = yaml_content.strip()
    
    
    try:
        parsed = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return {
            "error": "Gemini returned invalid YAML",
            "raw_response": yaml_content[:200],
            "parse_error": str(e)
        }
    
    
    template_path = os.path.join(TEMPLATES_DIR, f"{name}.yaml")
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    return {
        "template_name": name,
        "saved_to": template_path,
        "ai_model": "gemini-2.5-flash",
        "yaml_preview": yaml_content[:300] + "..." if len(yaml_content) > 300 else yaml_content,
        "parsed_successfully": True
    }

@mcp.tool
def run_template(template_name: str, path: str):
    """
    Apply AI-generated YAML to new PDF. Pure Python, zero AI cost.
    """
    
    template_path = os.path.join(TEMPLATES_DIR, f"{template_name}.yaml")
    if not os.path.exists(template_path):
        return {"error": f"Template '{template_name}' not found"}
    
    with open(template_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    
    
    existing = find_existing_pdf(path)
    if existing:
        pdf_id = existing
    else:
        result = ingest_pdf(path)
        pdf_id = result["pdf_id"]
    
    
    content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
    with open(content_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    
    signature = template.get("signature", {})
    keywords = signature.get("keywords", [])
    sig_page = signature.get("page", 1) - 1
    
    if sig_page < len(data["pages"]):
        page_text = data["pages"][sig_page]["text"]
        for keyword in keywords:
            if keyword not in page_text:
                 return {
                "status": "signature_mismatch",
                "expected": keyword,
                "found_in_page": page_text[:100] + "...",
                "suggestion": "This PDF may not match the template."
            }
   
     
    extracted = {}
    for rule_name, rule in template.get("extraction", {}).items():
        rule_type = rule.get("type")
        
        if rule_type == "table":
            page_num = rule.get("page")
            table_idx = rule.get("table_index", 0)
            
            
            table_data = extract_table(pdf_id, page_num, table_idx)
            extracted[rule_name] = table_data
        
        elif rule_type == "metadata":
            
            page_num = rule.get("page", 1)
            page_text = data["pages"][page_num - 1]["text"] if page_num <= len(data["pages"]) else ""
            
            fields = {}
            for field_name, field_rule in rule.get("fields", {}).items():
                pattern = field_rule.get("pattern", "")
                match = re.search(pattern, page_text)
                fields[field_name] = match.group(1) if match else "not_found"
            
            extracted[rule_name] = fields
    
    return {
        "status": "success",
        "template": template_name,
        "pdf_id": pdf_id,
        "validation": "signature_matched",
        "extracted": extracted
    }

@mcp.tool
def list_templates():
    """
    List all saved YAML templates.
    """
    results = []
    if not os.path.exists(TEMPLATES_DIR):
        return results
    
    for filename in os.listdir(TEMPLATES_DIR):
        if filename.endswith(".yaml"):
            path = os.path.join(TEMPLATES_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                template = yaml.safe_load(f)
                results.append({
                    "name": template.get("name", filename.replace(".yaml", "")),
                    "created_from": template.get("created_from_path", "unknown"),
                    "rules": len(template.get("extraction", {}))
                })
    return results



if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
    
