import logging
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils import convert_image_to_base64_and_test, test_with_base64_data

# Configure logging
logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Leaf Disease Detection API",
    version="1.1.0",
    description="Upload a leaf image and get an AI-powered disease "
                 "diagnosis with severity, symptoms, and treatment advice.",
)

# Allow browser-based frontends (e.g. a future React/HTML client) to call
# this API directly. Server-to-server calls (like the Streamlit app) are
# unaffected by CORS either way.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class Base64ImageRequest(BaseModel):
    image: str  # base64 string, with or without a data: prefix


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    """Raise HTTPException(400/413/415) for anything that isn't a reasonable image upload."""
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) / (1024*1024):.1f} MB). "
                   f"Max size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Please upload a JPEG, PNG, or WebP image.",
        )


@app.post("/disease-detection-file")
async def disease_detection_file(file: UploadFile = File(...)):
    """
    Detect diseases in a leaf image using direct file upload
    (multipart/form-data).
    """
    start = time.monotonic()
    try:
        logger.info("Received image file for disease detection: %s", file.filename)
        contents = await file.read()
        _validate_upload(file, contents)

        result = convert_image_to_base64_and_test(contents)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process image file")

        elapsed = time.monotonic() - start
        logger.info("Disease detection from file completed in %.2fs", elapsed)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        # Validation-style errors raised by the detector (bad/oversized image, etc.)
        logger.warning("Rejected image: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error in disease detection (file): %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/disease-detection-base64")
async def disease_detection_base64(payload: Base64ImageRequest):
    """
    Detect diseases in a leaf image supplied as base64-encoded JSON.
    Useful for clients that already have the image in memory (e.g. a
    browser canvas capture) and don't want to do a multipart upload.
    """
    start = time.monotonic()
    try:
        if not payload.image:
            raise HTTPException(status_code=400, detail="'image' field is empty")

        result = test_with_base64_data(payload.image)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to process image data")

        elapsed = time.monotonic() - start
        logger.info("Disease detection from base64 completed in %.2fs", elapsed)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Rejected image: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error in disease detection (base64): %s", e)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health():
    """Lightweight health check for uptime monitors / deployment platforms."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Leaf Disease Detection API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": {
            "disease_detection_file": "POST /disease-detection-file (multipart file upload)",
            "disease_detection_base64": "POST /disease-detection-base64 (JSON: {\"image\": \"<base64>\"})",
            "health": "GET /health",
        },
    }
