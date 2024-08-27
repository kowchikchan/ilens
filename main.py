from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from fire import process_video

# Define the request and response models
class FirePoints(BaseModel):
    fire: float

class Points(BaseModel):
    fire: Optional[FirePoints] = None

class FrameRequest(BaseModel):
    videoPath: str
    points: Points

class FireSeverity(BaseModel):
    severity: str

class FrameResponse(BaseModel):
    frameNo: str
    points: Dict[str, FireSeverity]

# Initialize FastAPI
app = FastAPI()

@app.post("/process-frame/")
async def process_frame(request: FrameRequest):
    video_path = request.videoPath
    output_dir = '/Users/helloabc/Documents/hai/captured_frames'  # Set your output directory path here

    # Process the video to get the frame and confidence
    frame, confidence = process_video(input_video_path=video_path, output_dir=output_dir, capture_interval=1)

    # If the video could not be opened or processed, raise an exception
    if frame is None and confidence is None:
        raise HTTPException(status_code=400, detail=f"Unable to process video at {video_path}. Please check the file path and format.")

    # If confidence is found, create the response
    if confidence is not None:
        severity = 'High' if confidence >= 0.5 else 'Low'
        response = FrameResponse(
            frameNo=video_path,
            points={"fire": FireSeverity(severity=severity)}
        )
    else:
        response = FrameResponse(
            frameNo=video_path,
            points={}
        )

    return response

# Example usage for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
