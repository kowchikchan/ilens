from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from io import BytesIO
from fire1 import process_image  # Import your image processing function

app = FastAPI()

# FastAPI request model
class Points(BaseModel):
    fire: int

class FrameRequest(BaseModel):
    frameNo: int
    points: Points

# FastAPI response model
class FireSeverity(BaseModel):
    severity: str

class FrameResponse(BaseModel):
    frameNo: int
    fire: FireSeverity
    confidence: float

@app.post("/process_fire", response_model=FrameResponse)
async def process_fire_endpoint(
        file: UploadFile = File(...),  # Default argument
        request: FrameRequest  # Non-default argument
):
    frameNo = request.frameNo
    points = request.points

    # Read the image file
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Error reading image")

    # Process the image
    confidence, severity = process_image(image, points.fire == 1)

    response = FrameResponse(
        frameNo=frameNo,
        fire=FireSeverity(severity=severity),
        confidence=confidence
    )

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
