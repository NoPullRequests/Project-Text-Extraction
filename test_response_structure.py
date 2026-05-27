from PIL import Image

from services.llm_service import GeminiProvider, _extract_response_text

prompt = "Return this JSON: {\"test\": \"hello world\", \"number\": 123}"
image = Image.open("uploads/invoice_sample.jpg")
provider = GeminiProvider()

response = provider._generate_content([prompt, image], "structure-test")
image.close()

response_text = _extract_response_text(response)
print("response.text:", repr(response_text))
print("Length:", len(response_text))
print("\nCandidates:", len(response.candidates))
print("First candidate:", response.candidates[0])
print("\nContent:", response.candidates[0].content)
print("\nParts:", response.candidates[0].content.parts)
print("\nFirst part:", response.candidates[0].content.parts[0])
print("\nFirst part text:", response.candidates[0].content.parts[0].text)
print("First part text length:", len(response.candidates[0].content.parts[0].text))
