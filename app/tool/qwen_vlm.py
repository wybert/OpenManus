
from typing import Optional
from app.tool.base import BaseTool, ToolResult
import requests
import base64

class QwenVLMTool(BaseTool):
    name: str = "qwen_vlm"
    description: str = "Analyzes remote sensing images using Qwen VLM API"
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to the image file"},
            "question": {"type": "string", "description": "Question about the image"}
        },
        "required": ["image_path", "question"]
    }

    async def execute(self, image_path: str, question: str) -> ToolResult:
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call Qwen VLM API (replace with actual endpoint and headers)
            response = requests.post(
                "YOUR_QWEN_API_ENDPOINT",
                headers={"Authorization": "YOUR_API_KEY"},
                json={
                    "image": encoded_image,
                    "question": question
                }
            )
            
            if response.status_code == 200:
                return ToolResult(output=response.json()["answer"])
            else:
                return ToolResult(error=f"API request failed: {response.status_code}")
                
        except Exception as e:
            return ToolResult(error=f"Error analyzing image: {str(e)}")
