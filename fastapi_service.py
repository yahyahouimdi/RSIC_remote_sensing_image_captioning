"""
FastAPI service for CNN+LSTM Image Captioning Model
Provides endpoints to generate captions for remote sensing images
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from pathlib import Path
from typing import Optional, List
import logging
from nltk.tokenize import word_tokenize
import nltk
import uvicorn

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL ARCHITECTURE - MUST MATCH TRAINING CODE
# ============================================================================

class EncoderCNN(nn.Module):
    """CNN encoder using ResNet50"""
    def __init__(self, embed_size=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, images):
        features = self.resnet(images).squeeze()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = self.fc(features)
        return features


class DecoderRNN_Improved(nn.Module):
    """LSTM decoder with dropout"""
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=3000, 
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = self.dropout(embeddings)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        outputs, _ = self.lstm(embeddings)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Image Captioning API",
    description="CNN+LSTM service for generating captions from remote sensing images",
    version="1.0.0"
)

# Add CORS middleware to allow requests from other platforms
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES FOR MODEL AND VOCAB
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model paths
MODEL_PATH = "best_model_improved.pth"
VOCAB_PATH = "vocab.pt"

# Model hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_CAPTION_LEN = 20

# Global variables
encoder = None
decoder = None
vocab = None
inv_vocab = None
vocab_size = 0
transform = None


def load_model():
    """Load the trained model and vocabulary"""
    global encoder, decoder, vocab, inv_vocab, vocab_size, transform
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            logger.info("Please ensure 'best_model_improved.pth' is in the same directory")
            return False
        
        # Load vocabulary
        if not os.path.exists(VOCAB_PATH):
            logger.warning(f"Vocab file not found at {VOCAB_PATH}")
            logger.info("Vocabulary will need to be loaded from model checkpoint")
        
        # Initialize models
        logger.info("Initializing model architecture...")
        
        # Load checkpoint to get vocab size info
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        vocab_size = checkpoint.get('vocab_size', 3000)  # Default vocab size
        
        # Initialize models
        encoder = EncoderCNN(embed_size=EMBED_SIZE).to(device)
        decoder = DecoderRNN_Improved(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            vocab_size=vocab_size,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        # Load weights
        logger.info("Loading model weights...")
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Set to evaluation mode
        encoder.eval()
        decoder.eval()
        
        # Load or create vocabulary
        if os.path.exists(VOCAB_PATH):
            vocab_data = torch.load(VOCAB_PATH)
            vocab = vocab_data['vocab']
            inv_vocab = vocab_data['inv_vocab']
        else:
            logger.warning("Creating minimal vocabulary - results may be limited")
            # Create basic vocab - in production you should load the full vocab
            vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
            inv_vocab = {v: k for k, v in vocab.items()}
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        logger.info(f"  Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


def generate_caption_greedy(image_tensor: torch.Tensor, max_len: int = MAX_CAPTION_LEN) -> str:
    """
    Generate caption using greedy decoding
    
    Args:
        image_tensor: Input image tensor [1, 3, 224, 224]
        max_len: Maximum caption length
    
    Returns:
        Generated caption string
    """
    with torch.no_grad():
        feature = encoder(image_tensor.to(device))
        word = torch.tensor([[vocab["<START>"]]]).to(device)
        caption = []
        
        for _ in range(max_len):
            output = decoder(feature, word)
            pred = output.argmax(2)[:, -1].item()
            
            if pred == vocab["<END>"]:
                break
            
            word_str = inv_vocab.get(pred, "<UNK>")
            if word_str not in ["<PAD>", "<START>", "<END>", "<UNK>"]:
                caption.append(word_str)
            
            word = torch.cat([word, torch.tensor([[pred]]).to(device)], 1)
    
    return " ".join(caption) if caption else "Unable to generate caption"


def generate_caption_beam_search(image_tensor: torch.Tensor, beam_width: int = 5, 
                                   max_len: int = MAX_CAPTION_LEN) -> str:
    """
    Generate caption using beam search decoding
    
    Args:
        image_tensor: Input image tensor [1, 3, 224, 224]
        beam_width: Beam width for search
        max_len: Maximum caption length
    
    Returns:
        Generated caption string
    """
    with torch.no_grad():
        feature = encoder(image_tensor.to(device))
        beams = [([vocab["<START>"]], 0.0)]
        
        for _ in range(max_len):
            new_beams = []
            
            for seq, score in beams:
                if seq[-1] == vocab["<END>"]:
                    new_beams.append((seq, score))
                    continue
                
                word = torch.tensor([seq]).to(device)
                output = decoder(feature, word)
                log_probs = torch.log_softmax(output[0, -1], dim=0)
                top_k_probs, top_k_indices = torch.topk(log_probs, min(beam_width, log_probs.size(0)))
                
                for prob, idx in zip(top_k_probs, top_k_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[-1] == vocab["<END>"] for seq, _ in beams):
                break
        
        best_seq = beams[0][0]
        caption = [inv_vocab.get(idx, "<UNK>") for idx in best_seq
                   if idx not in [vocab["<START>"], vocab["<END>"], 0]]
    
    return " ".join(caption) if caption else "Unable to generate caption"


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up... Loading model")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Image Captioning API",
        "description": "CNN+LSTM service for generating captions from remote sensing images",
        "version": "1.0.0",
        "status": "ready" if encoder and decoder else "not ready",
        "model": "CNN+LSTM (ResNet50 + LSTM)",
        "endpoints": {
            "/caption/greed": "POST - Generate caption using greedy decoding",
            "/caption/beam": "POST - Generate caption using beam search",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if encoder and decoder else "unhealthy",
        "device": str(device),
        "model_loaded": encoder is not None and decoder is not None,
        "vocab_loaded": vocab is not None
    }


@app.post("/caption/greedy", tags=["Caption Generation"])
async def generate_caption_greedy_endpoint(file: UploadFile = File(...)) -> dict:
    """
    Generate image caption using greedy decoding
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with generated caption
    """
    if encoder is None or decoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50)")
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Generate caption
        caption = generate_caption_greedy(image_tensor)
        
        return {
            "caption": caption,
            "method": "greedy",
            "image_size": image.size,
            "model": "CNN+LSTM",
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in greedy caption generation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/caption/beam", tags=["Caption Generation"])
async def generate_caption_beam_endpoint(
    file: UploadFile = File(...),
    beam_width: int = 5
) -> dict:
    """
    Generate image caption using beam search decoding
    
    Args:
        file: Image file (JPG, PNG, etc.)
        beam_width: Beam width for search (default: 5)
    
    Returns:
        JSON with generated caption
    """
    if encoder is None or decoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if beam_width < 1 or beam_width > 20:
        raise HTTPException(status_code=400, detail="Beam width must be between 1 and 20")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50)")
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Generate caption
        caption = generate_caption_beam_search(image_tensor, beam_width=beam_width)
        
        return {
            "caption": caption,
            "method": "beam_search",
            "beam_width": beam_width,
            "image_size": image.size,
            "model": "CNN+LSTM",
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in beam search caption generation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/caption/compare", tags=["Caption Generation"])
async def compare_methods(file: UploadFile = File(...)) -> dict:
    """
    Generate caption using both methods and compare
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with captions from both methods
    """
    if encoder is None or decoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50)")
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Generate both captions
        greedy_caption = generate_caption_greedy(image_tensor)
        beam_caption = generate_caption_beam_search(image_tensor, beam_width=5)
        
        return {
            "image_size": image.size,
            "captions": {
                "greedy": greedy_caption,
                "beam_search": beam_caption
            },
            "model": "CNN+LSTM",
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in comparison: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run with: python fastapi_service.py
    # Or with: uvicorn fastapi_service:app --reload
    
    logger.info("Starting Image Captioning FastAPI Service...")
    logger.info(f"Service will be available at: http://localhost:8000")
    logger.info(f"API docs at: http://localhost:8000/docs")
    logger.info(f"Interactive API explorer at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "fastapi_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
