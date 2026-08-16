from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from inference import load_model, generate_caption

app = FastAPI(title='TSFE Image Captioning API')

# ── Load model once at startup ────────────────────────────────────────────────
model, word2idx, idx2word, cfg = load_model('checkpoints')


@app.get('/')
def root():
    return {'status': 'running', 'model': 'TSFE SwinV2 + Transformer Decoder'}


@app.post('/caption')
async def caption_image(file: UploadFile = File(...)):
    # ── Validate file type ────────────────────────────────────────────────────
    if file.content_type not in ('image/jpeg', 'image/png', 'image/jpg'):
        return JSONResponse(
            status_code=400,
            content={'error': 'Only JPEG and PNG images are supported'}
        )

    # ── Read and decode image ─────────────────────────────────────────────────
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert('RGB')

    # ── Generate caption ──────────────────────────────────────────────────────
    caption = generate_caption(model, image, cfg, idx2word, beam_size=3)

    return {
        'caption':    caption,
        'filename':   file.filename,
        'image_size': image.size,
    }