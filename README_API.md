# Image Captioning API - FastAPI Service

A fast and efficient REST API service for generating captions from satellite/aerial images using a CNN+LSTM deep learning model. Built for remote sensing image captioning (RSICD dataset).

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd path/to/CNN+LSTM

# Install dependencies
pip install -r requirements.txt

# Make sure you have the trained model
# Ensure 'best_model_improved.pth' is in the same directory
```

### 2. Start the Service

```bash
# Option 1: Direct Python
python fastapi_service.py

# Option 2: Using uvicorn (with auto-reload for development)
uvicorn fastapi_service:app --reload --host 0.0.0.0 --port 8000

# Option 3: With custom port
uvicorn fastapi_service:app --host 0.0.0.0 --port 5000
```

The API will be available at: `http://localhost:8000`

## 📚 API Documentation

### Interactive API Explorer
Once the service is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### 1. Health Check
```
GET /health
```
Check if the model is loaded and service is running.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "vocab_loaded": true
}
```

#### 2. Generate Caption (Greedy Decoding)
```
POST /caption/greedy
```
Generate a caption using greedy decoding (fastest).

**Parameters:**
- `file` (FormData): The image file (JPG, PNG, etc.)

**Response:**
```json
{
  "caption": "a city street with cars and buildings",
  "method": "greedy",
  "image_size": [700, 700],
  "model": "CNN+LSTM",
  "success": true
}
```

#### 3. Generate Caption (Beam Search)
```
POST /caption/beam
```
Generate a caption using beam search decoding (higher quality, slower).

**Parameters:**
- `file` (FormData): The image file
- `beam_width` (Query, optional, default=5): Beam width (1-20)

**Response:**
```json
{
  "caption": "a large city with buildings and roads",
  "method": "beam_search",
  "beam_width": 5,
  "image_size": [700, 700],
  "model": "CNN+LSTM",
  "success": true
}
```

#### 4. Compare Both Methods
```
POST /caption/compare
```
Generate captions using both methods for comparison.

**Parameters:**
- `file` (FormData): The image file

**Response:**
```json
{
  "image_size": [700, 700],
  "captions": {
    "greedy": "a city street with cars",
    "beam_search": "a large city street with many cars"
  },
  "model": "CNN+LSTM",
  "success": true
}
```

## 💻 Usage Examples

### Python Client
```python
import requests
from pathlib import Path

# Service URL
API_URL = "http://localhost:8000"

# Option 1: Simple greedy caption
image_path = "path/to/image.jpg"
with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{API_URL}/caption/greedy", files=files)
    result = response.json()
    print(f"Caption: {result['caption']}")

# Option 2: Better quality beam search
with open(image_path, 'rb') as f:
    files = {'file': f}
    params = {'beam_width': 5}
    response = requests.post(f"{API_URL}/caption/beam", files=files, params=params)
    result = response.json()
    print(f"Caption: {result['caption']}")

# Option 3: Compare both methods
with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{API_URL}/caption/compare", files=files)
    result = response.json()
    print("Greedy:", result['captions']['greedy'])
    print("Beam:", result['captions']['beam_search'])
```

### cURL Commands
```bash
# Greedy caption
curl -X POST "http://localhost:8000/caption/greedy" \
  -H "accept: application/json" \
  -F "file=@image.jpg"

# Beam search with custom beam width
curl -X POST "http://localhost:8000/caption/beam?beam_width=10" \
  -H "accept: application/json" \
  -F "file=@image.jpg"

# Compare methods
curl -X POST "http://localhost:8000/caption/compare" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function generateCaption(imagePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    
    try {
        const response = await axios.post(
            'http://localhost:8000/caption/greedy',
            form,
            { headers: form.getHeaders() }
        );
        console.log('Caption:', response.data.caption);
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

generateCaption('path/to/image.jpg');
```

## 🔧 Configuration

### Model Parameters
Edit `fastapi_service.py` to adjust:

```python
EMBED_SIZE = 256           # Embedding size
HIDDEN_SIZE = 512          # LSTM hidden size
NUM_LAYERS = 2             # LSTM layers
DROPOUT = 0.5              # Dropout rate
MAX_CAPTION_LEN = 20       # Maximum caption length
```

### Device Selection
Automatically uses GPU if available, falls back to CPU.
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### CORS Settings
By default, allows requests from any origin. For production, modify:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourplatform.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

## 📊 Model Details

- **Encoder**: ResNet50 (pretrained on ImageNet)
- **Decoder**: LSTM with Dropout
- **Training Dataset**: RSICD (Remote Sensing Image Captioning Dataset)
- **Image Input Size**: 224x224 RGB
- **Vocabulary Size**: ~3000+ words
- **Max Caption Length**: 20 tokens

## 🐛 Troubleshooting

### Model file not found
```
Error: Model file not found at best_model_improved.pth
```
**Solution**: Ensure `best_model_improved.pth` is in the same directory as `fastapi_service.py`

### CUDA out of memory
```python
# In fastapi_service.py, force CPU usage:
device = torch.device("cpu")
```

### Port already in use
```bash
# Use a different port
uvicorn fastapi_service:app --host 0.0.0.0 --port 5000
```

### Import errors
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

## 🚀 Deployment Options

### Local Network
```bash
uvicorn fastapi_service:app --host 0.0.0.0 --port 8000
```
Access from any machine on the network at: `http://your-machine-ip:8000`

### Gunicorn (Production)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker fastapi_service:app --bind 0.0.0.0:8000
```

### Docker (Optional)
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 API Response Format

All successful responses follow this format:
```json
{
  "caption": "generated text description",
  "method": "greedy" | "beam_search",
  "image_size": [width, height],
  "model": "CNN+LSTM",
  "success": true
}
```

Error responses:
```json
{
  "detail": "Error description"
}
```

## 🔐 Security Notes

1. **File Upload Limitation**: Current implementation has no file size limit
   - Add in production: `max_file_size = 10 * 1024 * 1024  # 10MB`

2. **Input Validation**: Check minimum image size (50x50 pixels)

3. **Rate Limiting**: Consider adding for production deployment

4. **CORS**: Default allows all origins - restrict in production

## 📞 Integration with Other Platforms

### For Your Teammates

1. **Start the service**: `python fastapi_service.py`
2. **Share the URL**: `http://your-ip:8000`
3. **Use any of the examples above** to integrate into their platform

### Example Integration in Another Python Project
```python
# In your teammate's code
import requests

class ImageCaptioningClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def caption_image(self, image_path, method="greedy", beam_width=5):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'beam_width': beam_width} if method == "beam" else {}
            endpoint = f"{self.api_url}/caption/{method}"
            response = requests.post(endpoint, files=files, params=params)
            return response.json()

# Usage
client = ImageCaptioningClient("http://localhost:8000")
result = client.caption_image("image.jpg", method="beam", beam_width=5)
print(result['caption'])
```

## 📊 Performance Notes

- **Inference Time**:
  - Greedy: ~0.5-1 second per image
  - Beam Search: ~1-2 seconds per image
- **GPU**: Significantly faster with CUDA/GPU
- **Batch Processing**: Can be optimized for multiple images

## 📄 License

This project is part of your university coursework. Check with your instructors for usage and sharing guidelines.

## ❓ Questions?

For issues or questions:
1. Check the API docs at `http://localhost:8000/docs`
2. Review error messages in the terminal logs
3. Check the troubleshooting section above

---

**Version**: 1.0.0  
**Last Updated**: 2024
