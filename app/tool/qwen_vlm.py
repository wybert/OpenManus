
from typing import Optional
from app.tool.base import BaseTool, ToolResult
import base64
from openai import OpenAI
import os

class QwenVLMTool(BaseTool):
    name: str = "qwen_vlm"
    description: str = "Analyzes remote sensing images using Qwen VLM API"
    client: OpenAI = None
    parameters: dict = {
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to the image file"},
            "question": {"type": "string", "description": "Question about the image"}
        },
        "required": ["image_path", "question"]
    }

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    async def execute(self, image_path: str, question: str) -> ToolResult:
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call Qwen VLM API using OpenAI client
            completion = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }]
            )
            
            return ToolResult(output=completion.choices[0].message.content)
                
        except Exception as e:
            return ToolResult(error=f"Error analyzing image: {str(e)}")
