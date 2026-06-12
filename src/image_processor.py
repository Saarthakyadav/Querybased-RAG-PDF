import base64
import os
import io
from PIL import Image
from langchain_groq import ChatGroq


class ImageProcessor:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

        #  Initialize ONCE (performance critical)
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=self.api_key,
            timeout=10
        )

    def _process_image(self, image_bytes):
        """
        Resize and compress image for faster inference
        """
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((768, 768), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=65)

        return buffer.getvalue()

    def get_image_description(self, image_bytes):
        """
        Generate structured, retrieval-optimized description
        """

        #  Skip invalid images
        if not image_bytes or len(image_bytes) < 100:
            return None

        try:
            processed_bytes = self._process_image(image_bytes)

            image_base64 = base64.b64encode(processed_bytes).decode("utf-8")

            # Structured prompt (VERY IMPORTANT)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
Analyze this image from a document and return structured information:

- Type (chart/table/diagram/photo)
- Main topic
- Key elements (axes, labels, objects)
- Important values or text
- Keywords for retrieval

Keep it concise and factual.
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            response = self.llm.invoke(messages)

            # Limit size (important for embeddings)
            description = response.content.strip()[:500]

            return description

        except Exception as e:
            print(f"Image skipped: {str(e)}")
            return None