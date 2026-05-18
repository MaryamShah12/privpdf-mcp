import os
import json
import re
import uuid

import yaml
from google import genai

from config import DATA_DIR, TEMPLATES_DIR, ensure_data_dir, ensure_templates_dir
from extractor import extract_pdf
from store import find_existing_pdf, save_chunks_to_faiss, search_faiss

gemini = genai.Client()


def register_tools(mcp):
    """Attach all tools to the given FastMCP instance."""

    @mcp.tool
    def ingest_pdf(path: str):
        """Extract text/tables from a PDF and index it for search."""
        ensure_data_dir()

        pages_data, method = extract_pdf(path)

        pdf_id = str(uuid.uuid4())[:8]
        pdf_folder = os.path.join(DATA_DIR, pdf_id)
        os.makedirs(pdf_folder)

        total_tables = sum(len(p["tables"]) for p in pages_data)
        full_text = "".join(p["text"] for p in pages_data)

        record = {
            "pdf_id": pdf_id,
            "original_path": path,
            "extraction_method": method,
            "page_count": len(pages_data),
            "total_tables": total_tables,
            "pages": pages_data
        }

        with open(os.path.join(pdf_folder, "content.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        semantic_chunks = save_chunks_to_faiss(pdf_id, pages_data)

        return {
            "pdf_id": pdf_id,
            "page_count": len(pages_data),
            "tables_found": total_tables,
            "method": method,
            "semantic_chunks": semantic_chunks,
            "sample_text": full_text[:200] + "..." if len(full_text) > 200 else full_text
        }

    @mcp.tool
    def list_pdfs():
        """List all ingested PDFs."""
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
        """Semantic search over an ingested PDF."""
        chunks = search_faiss(pdf_id, question)
        if not chunks:
            return "No results found."
        return "Here are the most relevant sections:\n\n" + "\n\n---\n\n".join(chunks)

    @mcp.tool
    def extract_table(pdf_id: str, page_num: int, table_index: int = 0):
        """Extract a specific table from a PDF page as structured data."""
        content_path = os.path.join(DATA_DIR, pdf_id, "content.json")

        with open(content_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        target_page = next((p for p in data["pages"] if p["page_no"] == page_num), None)
        if target_page is None:
            return {"error": f"Page {page_num} not found"}

        tables = target_page.get("tables", [])
        if not tables:
            return {"error": "No tables on this page"}
        if table_index >= len(tables):
            return {"error": f"Only {len(tables)} table(s) exist (you asked for #{table_index})"}

        # Clean the raw table
        clean_table = []
        for row in tables[table_index]:
            clean_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
            clean_table.append(clean_row)

        headers = clean_table[0]
        result = [
            {headers[i]: (row[i] if i < len(row) else "") for i in range(len(headers))}
            for row in clean_table[1:]
        ]

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
        Analyse a PDF's structure with Gemini once and save the rules as a
        YAML template — future extractions need zero AI calls.
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

        response = gemini.models.generate_content(model="models/gemini-2.5-flash", contents=prompt)
        yaml_content = response.text.strip().removeprefix("```yaml").removesuffix("```").strip()

        try:
            yaml.safe_load(yaml_content)  # validate
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
        """Apply a saved YAML template to a new PDF — no AI cost."""
        template_path = os.path.join(TEMPLATES_DIR, f"{template_name}.yaml")
        if not os.path.exists(template_path):
            return {"error": f"Template '{template_name}' not found"}

        with open(template_path, "r", encoding="utf-8") as f:
            template = yaml.safe_load(f)

        # Ingest only if we haven't seen this PDF before
        existing = find_existing_pdf(path)
        pdf_id = existing if existing else ingest_pdf(path)["pdf_id"]

        content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
        with open(content_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Signature check
        signature = template.get("signature", {})
        sig_page_idx = signature.get("page", 1) - 1
        if sig_page_idx < len(data["pages"]):
            page_text = data["pages"][sig_page_idx]["text"]
            for keyword in signature.get("keywords", []):
                if keyword not in page_text:
                    return {
                        "status": "signature_mismatch",
                        "expected": keyword,
                        "found_in_page": page_text[:100] + "...",
                        "suggestion": "This PDF may not match the template."
                    }

        # Run extraction rules
        extracted = {}
        for rule_name, rule in template.get("extraction", {}).items():
            rule_type = rule.get("type")

            if rule_type == "table":
                extracted[rule_name] = extract_table(pdf_id, rule.get("page"), rule.get("table_index", 0))

            elif rule_type == "metadata":
                page_num = rule.get("page", 1)
                page_text = data["pages"][page_num - 1]["text"] if page_num <= len(data["pages"]) else ""
                fields = {}
                for field_name, field_rule in rule.get("fields", {}).items():
                    match = re.search(field_rule.get("pattern", ""), page_text)
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
        """List all saved YAML extraction templates."""
        results = []
        if not os.path.exists(TEMPLATES_DIR):
            return results
        for filename in os.listdir(TEMPLATES_DIR):
            if filename.endswith(".yaml"):
                with open(os.path.join(TEMPLATES_DIR, filename), "r", encoding="utf-8") as f:
                    template = yaml.safe_load(f)
                results.append({
                    "name": template.get("name", filename.replace(".yaml", "")),
                    "created_from": template.get("created_from_path", "unknown"),
                    "rules": len(template.get("extraction", {}))
                })
        return results