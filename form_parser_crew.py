from crewai.tools import BaseTool
from google.cloud import documentai_v1 as documentai
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field
load_dotenv()

# Pydantic schema for input validation
class DocumentAnalysisToolSchema(BaseModel):
    file_path: str

class DocumentAnalysisTool(BaseTool):
    name: str = "DocumentAnalysisTool"
    description: str = "Parses documents using Document AI and Gemini to extract structured data."
    args_schema = DocumentAnalysisToolSchema

    project_id: str = Field(...)
    location: str = Field(...)
    processor_id: str = Field(...)
    gemini_api_key: str = Field(...)
    creds_path: str = Field(...)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.creds_path
        genai.configure(api_key=self.gemini_api_key)

    def _get_normalized_bbox(self, poly):
        if not poly.normalized_vertices:
            return {}
        verts = poly.normalized_vertices
        return {
            "x1": round(verts[0].x, 4), "y1": round(verts[0].y, 4),
            "x2": round(verts[1].x, 4), "y2": round(verts[1].y, 4),
            "x3": round(verts[2].x, 4), "y3": round(verts[2].y, 4),
            "x4": round(verts[3].x, 4), "y4": round(verts[3].y, 4)
        }

    def _get_text(self, layout_or_anchor, full_text):
        if hasattr(layout_or_anchor, "text_anchor"):
            text_anchor = layout_or_anchor.text_anchor
        else:
            text_anchor = layout_or_anchor
        if not text_anchor.text_segments:
            return ""
        return "".join([
            full_text[segment.start_index:segment.end_index]
            for segment in text_anchor.text_segments
        ]).strip()

    def _get_layout_info(self, text_anchor, doc):
        if not text_anchor.text_segments:
            return {"page_number": None, "bounding_box": {}}
        for page in doc.pages:
            for token in page.tokens:
                if token.layout.text_anchor.text_segments:
                    token_seg = token.layout.text_anchor.text_segments[0]
                    anchor_seg = text_anchor.text_segments[0]
                    if token_seg.start_index == anchor_seg.start_index:
                        return {
                            "page_number": page.page_number,
                            "bounding_box": self._get_normalized_bbox(token.layout.bounding_poly)
                        }
        return {"page_number": None, "bounding_box": {}}

    def _process_document(self, file_path, mime_type="application/pdf"):
        client = documentai.DocumentProcessorServiceClient()
        name = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
        with open(file_path, "rb") as file:
            input_doc = file.read()
        raw_document = documentai.RawDocument(content=input_doc, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)
        return client.process_document(request=request).document

    def _extract_text_with_coords(self, document):
        result = []
        for page in document.pages:
            full_text = document.text
            for token in page.tokens:
                if not token.layout.text_anchor.text_segments:
                    continue
                seg = token.layout.text_anchor.text_segments[0]
                text = full_text[seg.start_index:seg.end_index]
                result.append({
                    "text": text.strip(),
                    "bounding_box": self._get_normalized_bbox(token.layout.bounding_poly),
                    "confidence": token.layout.confidence,
                    "page_number": page.page_number
                })
        return result

    def _extract_key_value_pairs(self, doc):
        kv_pairs = []
        full_text = doc.text
        for page in doc.pages:
            for field in page.form_fields:
                kv_pairs.append({
                    "field": self._get_text(field.field_name, full_text),
                    "value": self._get_text(field.field_value, full_text),
                    "key_confidence": field.field_name.confidence,
                    "value_confidence": field.field_value.confidence,
                    "page_number": page.page_number,
                    "key_bounding_box": self._get_normalized_bbox(field.field_name.bounding_poly),
                    "value_bounding_box": self._get_normalized_bbox(field.field_value.bounding_poly),
                })
        return kv_pairs

    def _extract_named_entities(self, doc):
        kv_pairs = []
        full_text = doc.text
        for entity in doc.entities:
            key = entity.type_
            value = self._get_text(entity.text_anchor, full_text)
            layout = self._get_layout_info(entity.text_anchor, doc)
            kv_pairs.append({
                "field": key,
                "value": value,
                "confidence": entity.confidence,
                "page_number": layout["page_number"],
                "bounding_box": layout["bounding_box"]
            })
            for prop in entity.properties:
                sub_key = f"{key}.{prop.type_}"
                sub_val = self._get_text(prop.text_anchor, full_text)
                sub_info = self._get_layout_info(prop.text_anchor, doc)
                kv_pairs.append({
                    "field": sub_key,
                    "value": sub_val,
                    "page_number": sub_info["page_number"],
                    "bounding_box": sub_info["bounding_box"],
                    "confidence": prop.confidence
                })
        return kv_pairs

    def _extract_tables(self, doc):
        tables = []
        for page in doc.pages:
            for table in page.tables:
                for row in list(table.header_rows) + list(table.body_rows):
                    row_data = []
                    for cell in row.cells:
                        row_data.append({
                            "text": self._get_text(cell.layout, doc.text),
                            "page_number": page.page_number,
                            "bounding_box": self._get_normalized_bbox(cell.layout.bounding_poly),
                            "confidence": cell.layout.confidence
                        })
                    tables.append(row_data)
        return tables

    def _call_gemini_for_extraction(self, text):
        prompt = f"""
You are a document understanding model. Extract only **personal details** from the given text.

Return only a JSON object in this exact format:

```json
{{
  "personal_details": {{
    "name": "Full Name",
    "phone": "",
    "email": "example@example.com",
    "PAN": "",
    "GSTIN": "",
    "address": "Full Address Here"
  }}
}}
If a field is not present, omit it.

Text:
{text}
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        try:
            cleaned = response.text.strip()
            if cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:-1])
            return json.loads(cleaned)
        except Exception as e:
            print("[Gemini ERROR]", e)
            print("Raw Gemini Response:", response.text)
            return {"personal_details": {}}

    def _run(self, file_path: str) -> dict:
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"❌ Invalid or missing file path: {file_path}")
        
        print(f"📂 Processing document: {file_path}")
        doc = self._process_document(file_path)
        text_coords = self._extract_text_with_coords(doc)
        all_text = " ".join([t["text"] for t in text_coords])
        gemini_data = self._call_gemini_for_extraction(all_text)
        tables = self._extract_tables(doc)

        output = {
            "personal_details": gemini_data.get("personal_details", {}),
            "text_with_coords": text_coords,
            "key_value_pairs": self._extract_key_value_pairs(doc),
            "named_entities": self._extract_named_entities(doc),
            "tables": tables
        }

        save_to = os.path.join(os.getcwd(), "output2.json")
        with open(save_to, "w") as f:
            json.dump(output, f, indent=2)
        print(f"✅ Output saved to: {save_to}")
        
        return {
        "status": "success",
        "file_saved": save_to,
        "fields_extracted": list(output.keys())
        }
