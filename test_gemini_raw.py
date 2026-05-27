from PIL import Image

from services.llm_service import GeminiProvider, _extract_response_text, _get_finish_reason

prompt = """You are analyzing an Indian TAX INVOICE or GST INVOICE. This is a business document showing goods/services sold.

Extract ONLY what is clearly visible. Return null for anything unclear. NEVER guess.

Return ONLY this JSON, nothing else:
{
  "document_type": "invoice",
  "vendor_name": "string or null",
  "vendor_gstin": "15-char string or null",
  "invoice_number": "string or null",
  "invoice_date": "DD-MM-YYYY or null",
  "buyer_name": "string or null",
  "buyer_gstin": "15-char string or null",
  "subtotal": "number (before tax) or null",
  "gst_amount": "number (CGST+SGST or IGST) or null",
  "total_amount": "number (final amount) or null",
  "currency": "INR",
  "line_items": []
}"""

image = Image.open("uploads/invoice_sample.jpg")
provider = GeminiProvider()

print("Calling Gemini...")
response = provider._generate_content([prompt, image], "raw-test")
image.close()

print("Response type:", type(response))
print("\nResponse text:")
response_text = _extract_response_text(response)
print(response_text)
print("\nLength:", len(response_text))

# Check finish reason and safety
print("\nFinish reason:", _get_finish_reason(response) or "N/A")
print("Safety ratings:", response.candidates[0].safety_ratings if response.candidates else "N/A")

# Try parsing it
import json
try:
    parsed = json.loads(response_text)
    print("\nParsed successfully!")
    print("Document type:", parsed.get('document_type'))
except Exception as e:
    print(f"\nParsing failed: {e}")
