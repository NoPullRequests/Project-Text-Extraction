import sys
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from services.llm_service import GeminiProvider
from PIL import Image

# Test Gemini provider directly
provider = GeminiProvider()

# Test with a simple image
image_path = "uploads/invoice_sample.jpg"

try:
    print("Testing Gemini extraction...")
    document = provider.extract_from_image(image_path, "invoice", "test-request-123")
    print(f"Success! Document type: {type(document).__name__}")
    print(f"Processing status: {document.processing_status}")
    print(f"Document type: {document.document_type}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
