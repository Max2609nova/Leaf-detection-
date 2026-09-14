import os
import io
import json
import base64
import logging
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from groq import Groq
from dotenv import load_dotenv

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:  # Pillow is optional but recommended
    _PIL_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DiseaseAnalysisResult:
    """
    Data class for storing comprehensive disease analysis results.

    This class encapsulates all the information returned from a leaf disease
    analysis, including detection status, disease identification, severity
    assessment, and treatment recommendations.

    Attributes:
        disease_detected (bool): Whether a disease was detected in the leaf image
        disease_name (Optional[str]): Name of the identified disease, None if healthy
        disease_type (str): Category of disease (fungal, bacterial, viral, pest, etc.)
    """
    disease_detected: bool
    disease_name: Optional[str]
    disease_type: str
    severity: str
    confidence: float
    symptoms: List[str]
    possible_causes: List[str]
    treatment: List[str]
    # NOTE: use a default_factory so the timestamp is generated fresh for
    # every analysis. A plain default (e.g. `= datetime.now()...isoformat()`)
    # is only evaluated ONCE when the class is defined, so every single
    # result would share the same stale timestamp from server startup.
    analysis_timestamp: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat())


class LeafDiseaseDetector:
    """
    Advanced Leaf Disease Detection System using AI Vision Analysis.

    This class provides comprehensive leaf disease detection capabilities using
    the Groq API with Llama Vision models. It can analyze leaf images to identify
    diseases, assess severity, and provide treatment recommendations. The system
    also validates that uploaded images contain actual plant leaves and rejects
    images of humans, animals, or other non-plant objects.

    The system supports base64 encoded images and returns structured JSON results
    containing disease information, confidence scores, symptoms, causes, and
    treatment suggestions.

    Features:
        - Image validation (ensures uploaded images contain plant leaves)
        - Multi-disease detection (fungal, bacterial, viral, pest, nutrient deficiency)
        - Severity assessment (mild, moderate, severe)
        - Confidence scoring (0-100%)
        - Symptom identification
        - Treatment recommendations
        - Robust error handling and response parsing
        - Invalid image type detection and rejection

    Attributes:
        MODEL_NAME (str): The AI model used for analysis
        DEFAULT_TEMPERATURE (float): Default temperature for response generation
        DEFAULT_MAX_TOKENS (int): Default maximum tokens for responses
        api_key (str): Groq API key for authentication
        client (Groq): Groq API client instance

    Example:
        >>> detector = LeafDiseaseDetector()
        >>> result = detector.analyze_leaf_image_base64(base64_image_data)
        >>> if result['disease_type'] == 'invalid_image':
        ...     print("Please upload a plant leaf image")
        >>> elif result['disease_detected']:
        ...     print(f"Disease detected: {result['disease_name']}")
        >>> else:
        ...     print("Healthy leaf detected")
    """

    MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 1024

    # Images larger than this (longest edge, in pixels) are downscaled before
    # being sent to the model. This keeps requests fast and cheap without any
    # meaningful loss of detection accuracy for leaf-disease symptoms.
    MAX_IMAGE_DIMENSION = 1280
    JPEG_QUALITY = 85
    # Hard cap on the raw upload size we're willing to process.
    MAX_INPUT_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self, api_key: Optional[str] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        """
        Initialize the Leaf Disease Detector with API credentials.

        Sets up the Groq API client and validates the API key from either
        the parameter or environment variables. Initializes logging for
        tracking analysis operations.

        Args:
            api_key (Optional[str]): Groq API key. If None, will attempt to
                                   load from GROQ_API_KEY environment variable.

        Raises:
            ValueError: If no valid API key is found in parameters or environment.

        Note:
            Ensure your .env file contains GROQ_API_KEY or pass it directly.
        """
        load_dotenv()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your environment or a "
                ".env file (see .env.example)."
            )
        self.client = Groq(api_key=self.api_key)

        # Allow model parameters to be overridden via constructor args or
        # environment variables, falling back to the class defaults.
        self.model_name = (
            model_name
            or os.environ.get("MODEL_NAME")
            or self.MODEL_NAME
        )
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.environ.get("MODEL_TEMPERATURE", self.DEFAULT_TEMPERATURE))
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.environ.get("MAX_COMPLETION_TOKENS", self.DEFAULT_MAX_TOKENS))
        )

        logger.info(
            "Leaf Disease Detector initialized (model=%s, temperature=%s, max_tokens=%s)",
            self.model_name, self.temperature, self.max_tokens
        )

    def _prepare_image(self, base64_image: str) -> str:
        """
        Validate and optimize an incoming base64 image before sending it to
        the model: enforce a size ceiling, downscale oversized images, and
        normalize the output to JPEG. Falls back to the original data
        untouched if Pillow isn't installed or the image can't be parsed
        (the model will still attempt analysis; format issues surface as a
        normal API error rather than crashing here).

        Args:
            base64_image (str): Raw base64 image data (no data: prefix)

        Returns:
            str: base64 image data, resized/normalized when possible
        """
        try:
            raw_bytes = base64.b64decode(base64_image, validate=False)
        except Exception:
            # Not valid base64 at all -- let the caller's validation catch it
            return base64_image

        if len(raw_bytes) > self.MAX_INPUT_BYTES:
            raise ValueError(
                f"Image is too large ({len(raw_bytes) / (1024*1024):.1f} MB). "
                f"Please upload an image under {self.MAX_INPUT_BYTES // (1024*1024)} MB."
            )

        if not _PIL_AVAILABLE:
            return base64_image

        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()  # confirms it's a real, undamaged image
            # Re-open after verify() (which leaves the file unusable for further ops)
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img = img.convert("RGB")
                if max(img.size) > self.MAX_IMAGE_DIMENSION:
                    img.thumbnail(
                        (self.MAX_IMAGE_DIMENSION, self.MAX_IMAGE_DIMENSION),
                        Image.LANCZOS,
                    )
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=self.JPEG_QUALITY)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except ValueError:
            raise
        except Exception as e:
            # Not a valid/openable image -- surface a clear, specific error
            # instead of silently forwarding garbage to the model.
            raise ValueError(f"Uploaded file is not a valid image: {e}")

    def create_analysis_prompt(self) -> str:
        """
        Create the standardized analysis prompt for the AI model.

        Generates a comprehensive prompt that instructs the AI model to analyze
        leaf images for diseases and return structured JSON results. The prompt
        specifies the required output format and analysis criteria.

        Returns:
            str: Formatted prompt string with instructions for disease analysis
                 and JSON schema specification.

        Note:
            The prompt ensures consistent output formatting across all analyses
            and includes all necessary fields for comprehensive disease assessment.
        """
        return """IMPORTANT: First determine if this image contains a plant leaf or vegetation. If the image shows humans, animals, objects, buildings, or anything other than plant leaves/vegetation, return the "invalid_image" response format below.

        If this is a valid leaf/plant image, analyze it for diseases and return the results in JSON format.
        
        Please identify:
        1. Whether this is actually a leaf/plant image
        2. Disease name (if any)
        3. Disease type/category or invalid_image
        4. Severity level (mild, moderate, severe)
        5. Confidence score (0-100%)
        6. Symptoms observed
        7. Possible causes
        8. Treatment recommendations

        For NON-LEAF images (humans, animals, objects, or not detected as leaves, etc.), return this format:
        {
            "disease_detected": false,
            "disease_name": null,
            "disease_type": "invalid_image",
            "severity": "none",
            "confidence": 95,
            "symptoms": ["This image does not contain a plant leaf"],
            "possible_causes": ["Invalid image type uploaded"],
            "treatment": ["Please upload an image of a plant leaf for disease analysis"]
        }
        
        For VALID LEAF images, return this format:
        {
            "disease_detected": true/false,
            "disease_name": "name of disease or null",
            "disease_type": "fungal/bacterial/viral/pest/nutrient deficiency/healthy",
            "severity": "mild/moderate/severe/none",
            "confidence": 85,
            "symptoms": ["list", "of", "symptoms"],
            "possible_causes": ["list", "of", "causes"],
            "treatment": ["list", "of", "treatments"]
        }"""

    def analyze_leaf_image_base64(self, base64_image: str,
                                  temperature: float = None,
                                  max_tokens: int = None) -> Dict:
        """
        Analyze base64 encoded image data for leaf diseases and return JSON result.

        First validates that the image contains a plant leaf. If the image shows
        humans, animals, objects, or other non-plant content, returns an 
        'invalid_image' response. For valid leaf images, performs disease analysis.

        Args:
            base64_image (str): Base64 encoded image data (without data:image prefix)
            temperature (float, optional): Model temperature for response generation
            max_tokens (int, optional): Maximum tokens for response

        Returns:
            Dict: Analysis results as dictionary (JSON serializable)
                 - For invalid images: disease_type will be 'invalid_image'
                 - For valid leaves: standard disease analysis results

        Raises:
            Exception: If analysis fails
        """
        try:
            logger.info("Starting analysis for base64 image data")

            # Validate base64 input
            if not isinstance(base64_image, str):
                raise ValueError("base64_image must be a string")

            if not base64_image:
                raise ValueError("base64_image cannot be empty")

            # Clean base64 string (remove data URL prefix if present)
            if base64_image.startswith('data:'):
                base64_image = base64_image.split(',', 1)[1]

            # Validate, downscale, and normalize the image before sending it
            # to the model (raises ValueError for bad/oversized input).
            base64_image = self._prepare_image(base64_image)

            # Prepare request parameters
            temperature = temperature if temperature is not None else self.temperature
            max_tokens = max_tokens or self.max_tokens

            # Make API request
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.create_analysis_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )

            logger.info("API request completed successfully")
            result = self._parse_response(
                completion.choices[0].message.content)

            # Return as dictionary for JSON serialization
            return result.__dict__

        except Exception as e:
            logger.error(f"Analysis failed for base64 image data: {str(e)}")
            raise

    def _parse_response(self, response_content: str) -> DiseaseAnalysisResult:
        """
        Parse and validate API response

        Args:
            response_content (str): Raw response from API

        Returns:
            DiseaseAnalysisResult: Parsed and validated results
        """
        try:
            # Clean up response - remove markdown code blocks if present
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace(
                    '```json', '').replace('```', '').strip()
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.replace('```', '').strip()

            # Parse JSON
            disease_data = json.loads(cleaned_response)
            logger.info("Response parsed successfully as JSON")

            # Validate required fields and create result object
            return DiseaseAnalysisResult(
                disease_detected=bool(
                    disease_data.get('disease_detected', False)),
                disease_name=disease_data.get('disease_name'),
                disease_type=disease_data.get('disease_type', 'unknown'),
                severity=disease_data.get('severity', 'unknown'),
                confidence=float(disease_data.get('confidence', 0)),
                symptoms=disease_data.get('symptoms', []),
                possible_causes=disease_data.get('possible_causes', []),
                treatment=disease_data.get('treatment', [])
            )

        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse as JSON, attempting to extract JSON from response")

            # Try to find JSON in the response using regex
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    disease_data = json.loads(json_match.group())
                    logger.info("JSON extracted and parsed successfully")

                    return DiseaseAnalysisResult(
                        disease_detected=bool(
                            disease_data.get('disease_detected', False)),
                        disease_name=disease_data.get('disease_name'),
                        disease_type=disease_data.get(
                            'disease_type', 'unknown'),
                        severity=disease_data.get('severity', 'unknown'),
                        confidence=float(disease_data.get('confidence', 0)),
                        symptoms=disease_data.get('symptoms', []),
                        possible_causes=disease_data.get(
                            'possible_causes', []),
                        treatment=disease_data.get('treatment', [])
                    )
                except json.JSONDecodeError:
                    pass

            # If all parsing attempts fail, log the raw response and raise error
            logger.error(
                f"Could not parse response as JSON. Raw response: {response_content}")
            raise ValueError(
                f"Unable to parse API response as JSON: {response_content[:200]}...")


def main():
    """Main execution function for testing"""
    try:
        # Example usage
        detector = LeafDiseaseDetector()
        print("Leaf Disease Detector (minimal version) initialized successfully!")
        print("Use analyze_leaf_image_base64() method with base64 image data.")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
