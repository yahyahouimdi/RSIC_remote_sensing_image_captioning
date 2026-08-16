import json
import torch
import torchvision.transforms as T
from PIL import Image
from model import TSFE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(ckpt_dir='checkpoints'):
    # ── Load vocabulary ───────────────────────────────────────────────────────
    with open(f'{ckpt_dir}/word2idx.json') as f:
        word2idx = json.load(f)
    with open(f'{ckpt_dir}/idx2word.json') as f:
        idx2word = {int(k): v for k, v in json.load(f).items()}
    with open(f'{ckpt_dir}/config.json') as f:
        cfg = json.load(f)

    # ── Convert config values back to correct types ───────────────────────────
    int_keys = ['vocab_size', 'embed_dim', 'local_feat_dim', 'global_feat_dim',
                'dec_layers', 'dec_d_model', 'dec_nhead', 'dec_dim_feedfwd',
                'num_heads', 'n_local_attn', 'max_seq_len', 'batch_size',
                'accum_steps', 'ft_epochs', 'train_epochs', 'beam_size',
                'sos_id', 'eos_id', 'pad_id', 'img_size']
    for k in int_keys:
        if k in cfg and cfg[k] is not None:
            cfg[k] = int(float(cfg[k]))

    # ── Build model and load weights ──────────────────────────────────────────
    import numpy as np
    emb_matrix = np.zeros((cfg['vocab_size'], cfg['embed_dim']), dtype='float32')

    model = TSFE(cfg, emb_matrix).to(DEVICE)
    ckpt  = torch.load(
        f'{ckpt_dir}/tsfe_best.pth',
        map_location=DEVICE,
        weights_only=False
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'✅ Model loaded — vocab={cfg["vocab_size"]}  device={DEVICE}')
    return model, word2idx, idx2word, cfg


def preprocess_image(image: Image.Image, img_size=256):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)


def ids_to_str(ids, idx2word):
    words = []
    for i in ids:
        w = idx2word.get(int(i), '<unk>')
        if w in ('<eos>', '<pad>'): break
        if w == '<sos>': continue
        words.append(w)
    return ' '.join(words) if words else '<empty>'


def generate_caption(model, image: Image.Image, cfg, idx2word, beam_size=3):
    img_tensor = preprocess_image(image, cfg['img_size'])
    with torch.no_grad():
        pred_ids = model.caption(
            img_tensor,
            max_len=cfg['max_seq_len'],
            beam_size=beam_size
        )
    return ids_to_str(pred_ids, idx2word)