# PrivatePDF MCP Server

Give Claude Desktop the ability to read, query, and extract structured data from any PDF — entirely on your own machine.

Built with FastMCP, pdfplumber, pytesseract, FAISS, and Gemini API.


## What It Does

Most PDF tools send your documents to the cloud. PrivatePDF doesn't. It runs as a local MCP server — Claude Desktop talks to it directly, your files never leave your environment (except a small structural summary sent to Gemini once per document type for template generation).

The clever part: **generate a reusable YAML template from one PDF, then extract all future PDFs of the same type with zero AI calls.** Pure local logic, no API cost.


## Features

- Auto-detects PDF type — text-based PDFs use pdfplumber; scanned/image PDFs fall back to Tesseract OCR
- Semantic search via FAISS so Claude can find relevant sections across a document
- Structured table extraction as clean JSON
- Template system — Gemini analyses structure once, saves as YAML, all future extractions are free
- Everything runs locally — no document content uploaded anywhere


## MCP Tools

| Tool | What it does |
|------|-------------|
| `ingest_pdf(path)` | Extract text/tables, build FAISS index, store locally |
| `query_pdf(pdf_id, question)` | Semantic search over an ingested PDF |
| `extract_table(pdf_id, page_num, table_index)` | Pull a specific table as structured JSON |
| `generate_template(pdf_id, name)` | AI analyses structure once, saves as reusable YAML |
| `run_template(template_name, path)` | Apply saved template to new PDF — no AI call |
| `list_pdfs()` | Show all ingested PDFs |
| `list_templates()` | Show all saved templates |


## Setup

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on PATH
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) on PATH (Windows)
- A [Gemini API key](https://aistudio.google.com) (free tier works)

### Install

```bash
git clone https://github.com/MaryamShah12/privpdf-mcp
cd privpdf-mcp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Add your key to `.env`:
```
GOOGLE_API_KEY=your_key_here
```


## Connect to Claude Desktop

Find your config at `C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "privpdf": {
      "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\main.py"],
      "env": {
        "GOOGLE_API_KEY": "your_key_here"
      }
    }
  }
}
```

Restart Claude Desktop — the tools appear automatically.


## Example

```
You:    Ingest this PDF: C:\docs\invoice_jan.pdf
Claude: Done. 1 page, 1 table, 12 semantic chunks. ID: 8cbdfd56

You:    What is the total amount due?
Claude: $11,151.00 (subtotal $10,325 + 8% tax)

You:    Save this as a template called acme_invoice
Claude: Template saved. Future invoices extracted with zero AI calls.

You:    Run acme_invoice on C:\docs\invoice_feb.pdf
Claude: Extracted. Signature matched. Done instantly.
```


## Notes

- `data/` and `templates/` directories are created automatically on first run
- Re-ingesting the same file path is safe — duplicates are skipped
- Gemini is only called during `generate_template()` — everything else runs offline