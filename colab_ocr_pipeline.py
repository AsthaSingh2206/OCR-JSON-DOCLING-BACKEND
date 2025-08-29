# Cell 1: Install dependencies
import sys
import os
import sqlite3
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from paddleocr import PaddleOCR
from PyPDF2 import PdfReader
from huggingface_hub import snapshot_download
import time

# Cell 2: PDF Upload and OCR to JSON
# Helper: create pipeline options
# Step 1: Download PaddleOCR ONNX models once at script start
print("Downloading PaddleOCR models for ONNX backend...")
download_path = snapshot_download(repo_id="SWHL/RapidOCR")
det_model_path = os.path.join(download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx")
rec_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx")
cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
print("PaddleOCR models downloaded and paths set.")

def create_pipeline_options(ocr_backend: str, accelerator_device: AcceleratorDevice):
    ocr_backend = ocr_backend.lower()
    if ocr_backend == 'easy':
        ocr_options = EasyOcrOptions()
    elif ocr_backend == 'rapid':
        ocr_options = RapidOcrOptions()
    elif ocr_backend == 'paddle':
        # PaddleOCR integrated via RapidOcrOptions with ONNX models
        ocr_options = RapidOcrOptions(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )
    else:
        raise ValueError(f"Unsupported OCR backend: {ocr_backend}")
    accel_options = AcceleratorOptions(device=accelerator_device)
    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
        accelerator_options=accel_options,
    )
    return pipeline_options

from typing import Any, Dict, List, Union

def process_pdf(
    pdf_path: str,
    output_dir: str,
    ocr_backend: str,
    accelerator_device: AcceleratorDevice
) -> (str, Any):
    pipeline_options = create_pipeline_options(ocr_backend, accelerator_device)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    start_time = time.time()
    result = converter.convert(pdf_path)
    end_time = time.time()
    doc = result.document
    doc_dict = doc.model_dump()
    # Page count
    page_count = len(PdfReader(pdf_path).pages)
    # Text segmentation into sections
    texts = [t for t in doc_dict.get("texts", [])]
    texts_sorted = sorted(
        texts,
        key=lambda t: (
            t.get("prov", [{}])[0].get("page_no", 0),
            t.get("prov", [{}])[0].get("bbox", {}).get("t", 0)
        )
    )
    GAP_THRESHOLD = 10
    FONT_SIZE_THRESHOLD = 1.2
    sections: List[str] = []
    curr_sec: List[str] = []
    prev_bottom = None
    prev_font = None

    # Build 'texts' list for output
    texts_output: List[Dict[str, Any]] = []
    for t in texts_sorted:
        entry = {
            "page_num": t.get("prov", [{}])[0].get("page_no", 0),
            "text": t.get("text", "").strip()
        }
        texts_output.append(entry)

    # Helpers
    def normalize(val: Any) -> Any:
        if isinstance(val, str):
            v = val.replace(",", "")
            try:
                return float(v) if "." in v else int(v)
            except:
                return val
        return val

    def clean_rows(rows: List[List[Any]]) -> List[List[Any]]:
        return [row for row in rows if any(str(c).strip() for c in row)]

    # Build both table outputs
    tables_by_row_key: List[Dict[str, Union[Any, List[Any]]]] = []
    tables_by_columns: List[Dict[str, List[Any]]] = []

    def table_to_records(rows):
        if not rows:
            return []
        headers = [h if h else f"col_{i}" for i, h in enumerate(rows[0])]
        data_rows = rows[1:]
        output = []
        for row in data_rows:
            record = {}
            for i, header in enumerate(headers):
                record[header] = row[i] if i < len(row) else ""
            output.append(record)
        return output

    def flatten_key_value_table(rows):
        if len(rows) < 2:
            return None
        if len(rows[0]) in (2, 3):
            result = {}
            for row in rows:
                key = str(row[0]).strip() if len(row) > 0 else ""
                if not key or key.lower() in ["", "total", "subtotal"]:
                    continue
                value = row[1] if len(row) > 1 else ""
                result[key] = value
            return result
        return None

    for table in doc_dict.get("tables", []):
        # Extract raw grid
        raw: List[List[Any]] = []
        for row in table["data"]["grid"]:
            cells = [ normalize(cell.get("text","").strip()) if cell else "" for cell in row ]
            raw.append(cells)
        cleaned = clean_rows(raw)
        if not cleaned:
            tables_by_row_key.append({})
            tables_by_columns.append({})
            continue

        flattened = flatten_key_value_table(cleaned)

        if flattened:
            tables_by_columns.append(flattened)
        else:
            tables_by_columns.append(table_to_records(cleaned))

        # Row-keyed extraction: first cell as key, rest as value(s)
        row_keyed: Dict[str, Union[Any, List[Any]]] = {}

        for row in cleaned:
            if not row or not str(row[0]).strip():
                continue

            key = str(row[0]).strip()
            values = row[1:]

            if len(values) == 1:
                row_keyed[key] = values[0]
            else:
                row_keyed[key] = values

        tables_by_row_key.append(row_keyed)

    # Metadata
    metadata = doc_dict.get("metadata", {})
    metadata.update({
        "source_file": pdf_path,
        "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "page_count": page_count,
        "ocr_backend": ocr_backend,
        "accelerator_device": accelerator_device.name,
        "execution_time_sec": end_time - start_time,
    })

    output = {
        "metadata": metadata,
        "tables_rowwise": tables_by_row_key,
        "tables_columnwise": tables_by_columns,
        "texts": texts_output,
    }

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(output_dir, f"{base}_{ocr_backend}_refined_output.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Processed '{pdf_path}' with OCR '{ocr_backend}' on {accelerator_device.name}, saved: {out_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    return out_path, output
