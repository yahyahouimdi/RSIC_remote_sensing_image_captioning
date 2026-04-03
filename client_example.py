"""
Example Client for Image Captioning API
Demonstrates how to use the FastAPI service
"""

import requests
import json
from pathlib import Path
from typing import Optional, Dict
import time


class ImageCaptioningClient:
    """Client for communicating with the Image Captioning API"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            api_url: Base URL of the API (default: localhost:8000)
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the API is running and model is loaded"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": f"Cannot connect to API at {self.api_url}"}
        except Exception as e:
            return {"error": str(e)}
    
    def caption_image_greedy(self, image_path: str) -> Dict:
        """
        Generate caption using greedy decoding (fast)
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Response dictionary with caption and metadata
        """
        return self._send_request(image_path, "/caption/greedy")
    
    def caption_image_beam(self, image_path: str, beam_width: int = 5) -> Dict:
        """
        Generate caption using beam search (better quality)
        
        Args:
            image_path: Path to the image file
            beam_width: Beam width for search (1-20, default: 5)
        
        Returns:
            Response dictionary with caption and metadata
        """
        return self._send_request(
            image_path, 
            "/caption/beam",
            params={"beam_width": beam_width}
        )
    
    def caption_image_compare(self, image_path: str) -> Dict:
        """
        Generate captions using both methods for comparison
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Response dictionary with both captions
        """
        return self._send_request(image_path, "/caption/compare")
    
    def _send_request(self, image_path: str, endpoint: str, 
                     params: Optional[Dict] = None) -> Dict:
        """
        Internal method to send request to API
        
        Args:
            image_path: Path to image file
            endpoint: API endpoint
            params: Optional query parameters
        
        Returns:
            Response dictionary
        """
        try:
            # Validate image file exists
            image_file = Path(image_path)
            if not image_file.exists():
                return {"error": f"Image file not found: {image_path}", "success": False}
            
            # Check file is a valid image
            valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
            if image_file.suffix.lower() not in valid_extensions:
                return {"error": f"Invalid image format: {image_file.suffix}", "success": False}
            
            # Send request
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(
                    f"{self.api_url}{endpoint}",
                    files=files,
                    params=params,
                    timeout=30
                )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            return {"error": "Request timed out", "success": False}
        except requests.exceptions.ConnectionError:
            return {"error": f"Cannot connect to API at {self.api_url}", "success": False}
        except requests.exceptions.HTTPError as e:
            return {"error": f"API error: {e.response.json().get('detail', str(e))}", "success": False}
        except Exception as e:
            return {"error": f"Error: {str(e)}", "success": False}
    
    def close(self):
        """Close the session"""
        self.session.close()


def print_result(result: Dict, method: str = ""):
    """Pretty print the API response"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n{'='*60}")
    if method:
        print(f"Method: {method}")
    print(f"{'='*60}")
    
    if 'caption' in result:
        print(f"Caption: {result['caption']}")
    
    if 'captions' in result:
        print(f"Greedy:       {result['captions']['greedy']}")
        print(f"Beam Search:  {result['captions']['beam_search']}")
    
    if 'image_size' in result:
        print(f"Image Size:   {result['image_size']}")
    
    if 'method' in result:
        print(f"Decoding:     {result['method']}")
    
    if 'beam_width' in result:
        print(f"Beam Width:   {result['beam_width']}")
    
    print(f"{'='*60}\n")


def main():
    """Example usage of the Image Captioning Client"""
    
    print("Image Captioning API - Client Example")
    print("=" * 60)
    
    # Initialize client
    client = ImageCaptioningClient("http://localhost:8000")
    
    # Check API health
    print("\n1. Checking API Health...")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model Loaded: {health.get('model_loaded', False)}")
    print(f"   Device: {health.get('device', 'unknown')}")
    
    if not health.get('model_loaded'):
        print("\n❌ Model not loaded! Make sure to start the FastAPI service first:")
        print("   python fastapi_service.py")
        return
    
    # Example image paths (modify these with your actual image paths)
    example_images = [
        "image.jpg",
        "sample.png",
        "test_image.jpg"
    ]
    
    # Find first existing image for testing
    test_image = None
    for img_path in example_images:
        if Path(img_path).exists():
            test_image = img_path
            break
    
    if not test_image:
        print("\n📝 No test images found.")
        print("   Usage: Place an image file (JPG, PNG) in the current directory")
        print("   Then modify example_images in this script to test")
        
        # Show example usage
        print("\n📚 Example Usage:")
        print("-" * 60)
        print("""
# Create client
client = ImageCaptioningClient("http://localhost:8000")

# Method 1: Fast greedy decoding
result = client.caption_image_greedy("path/to/image.jpg")
print(result['caption'])

# Method 2: Better quality beam search
result = client.caption_image_beam("path/to/image.jpg", beam_width=5)
print(result['caption'])

# Method 3: Compare both methods
result = client.caption_image_compare("path/to/image.jpg")
print("Greedy:", result['captions']['greedy'])
print("Beam:", result['captions']['beam_search'])
        """)
        return
    
    print(f"\n2. Testing with image: {test_image}")
    
    # Test greedy decoding
    print("\n3. Generating caption with GREEDY decoding...")
    start = time.time()
    result_greedy = client.caption_image_greedy(test_image)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print_result(result_greedy, "Greedy")
    
    # Test beam search
    print("\n4. Generating caption with BEAM SEARCH decoding...")
    start = time.time()
    result_beam = client.caption_image_beam(test_image, beam_width=5)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print_result(result_beam, "Beam Search (width=5)")
    
    # Compare methods
    print("\n5. Comparing both methods...")
    result_compare = client.caption_image_compare(test_image)
    print_result(result_compare, "Comparison")
    
    # Show timestamps
    print("\n6. Performance Summary:")
    print(f"   Greedy Speed: {elapsed:.2f}s (faster)")
    print(f"   Beam Search: ~{elapsed*2:.2f}s (better quality)")
    
    # Clean up
    client.close()
    
    print("\n✅ Example completed successfully!")
    print("\nFor more information, see README_API.md")


if __name__ == "__main__":
    main()
