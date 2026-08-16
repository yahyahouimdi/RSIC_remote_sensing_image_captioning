import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ═══════════════════════════════════════════════════════════════════════════════
# 6a. Adaptive Multi-Scale Feature Fusion (AMFF)
#     Enhancement 1: SwinV2-Base feature channels F2=256, F3=512, F4=1024
#     (vs Swin-Base: F2=192, F3=384, F4=768)
#     All other logic identical to paper Eqs 1–5.
# ═══════════════════════════════════════════════════════════════════════════════

class AMFF(nn.Module):
    """
    Adaptive Multi-Scale Feature Fusion.
    SwinV2-Base: F2=256ch/32×32, F3=512ch/16×16, F4=1024ch/8×8
    After interpolation to 8×8 and concat → 256+512+1024 = 1792 ch
    """
    def __init__(self, cfg):
        super().__init__()
        # SwinV2-Base channels (Enhancement 1)
        concat_ch = 256 + 512 + 1024         # 1792

        # SENet channel-attention (paper Eq. 3)
        r = 16
        self.se_avg = nn.AdaptiveAvgPool2d(1)
        self.se_fc  = nn.Sequential(
            nn.Linear(concat_ch, concat_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(concat_ch // r, concat_ch, bias=False),
            nn.Sigmoid()
        )

        # Project to local_feat_dim (2048)
        self.proj = nn.Sequential(
            nn.Conv2d(concat_ch, cfg['local_feat_dim'], 1, bias=False),
            nn.BatchNorm2d(cfg['local_feat_dim']),
            nn.ReLU(inplace=True)
        )

        # Trainable scalars w1, w2 (paper Eq. 4)
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))

        # Project F4 (1024ch) to local_feat_dim for residual path
        self.f4_proj = nn.Sequential(
            nn.Conv2d(1024, cfg['local_feat_dim'], 1, bias=False),
            nn.BatchNorm2d(cfg['local_feat_dim']),
            nn.ReLU(inplace=True)
        )

        # Global feature alignment MLP
        self.global_proj = nn.Sequential(
            nn.Linear(cfg['local_feat_dim'], cfg['local_feat_dim'] // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg['local_feat_dim'] // 2, cfg['global_feat_dim'])
        )
        self.cfg = cfg

    def forward(self, f2, f3, f4):
        h, w = f4.shape[2], f4.shape[3]          # 8×8 for 256-input SwinV2
        f2_up = F.interpolate(f2, size=(h, w), mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=(h, w), mode='bilinear', align_corners=False)
        F_cat = torch.cat([f2_up, f3_up, f4], dim=1)  # (B, 1792, h, w)

        s = self.se_avg(F_cat).flatten(1)
        s = self.se_fc(s).unsqueeze(-1).unsqueeze(-1)
        Fs = F_cat * s
        Fs = self.proj(Fs)                          # (B, local_feat_dim, h, w)

        F4p = self.f4_proj(f4)
        Vlocal_map = self.w1 * F4p + self.w2 * Fs

        gap = Vlocal_map.mean(dim=[2, 3])
        Vglobal = self.global_proj(gap)             # (B, embed_dim=300)

        B, C, H, W = Vlocal_map.shape
        Vlocal = Vlocal_map.flatten(2).permute(0, 2, 1)  # (B, H*W, local_feat_dim)
        return Vlocal, Vglobal, H, W
 # ═══════════════════════════════════════════════════════════════════════════════
#                                LFSE fixed
# ═══════════════════════════════════════════════════════════════════════════════

class LFSE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C  = cfg['local_feat_dim']  # 2048
        nh = cfg['num_heads']       # 8
        N  = cfg['n_local_attn']    # 8

        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)

        self.mha_h = nn.MultiheadAttention(C, nh, batch_first=True, dropout=0.1)
        self.mha_v = nn.MultiheadAttention(C, nh, batch_first=True, dropout=0.1)

        self.dwconv = nn.Sequential(
            nn.Conv2d(3 * C, 3 * C, 3, padding=1, groups=3 * C, bias=False),
            nn.BatchNorm2d(3 * C)
        )
        self.pw_conv = nn.Sequential(
            nn.Conv2d(3 * C, C, 1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.w_local = nn.Parameter(torch.randn(N, C) * 0.01)
        self.N = N
        self.C = C

    def forward(self, Vlocal, H, W):
        B, HW, C = Vlocal.shape
        assert HW == H * W

        q = self.q_proj(Vlocal)
        k = self.k_proj(Vlocal)
        v = self.v_proj(Vlocal)

        q_2d = q.view(B, H, W, C)
        k_2d = k.view(B, H, W, C)
        v_2d = v.view(B, H, W, C)

        q_h = q_2d.reshape(B * H, W, C)
        k_h = k_2d.reshape(B * H, W, C)
        v_h = v_2d.reshape(B * H, W, C)
        Va_h, _ = self.mha_h(q_h, k_h, v_h)
        Va_h = Va_h.reshape(B, H, W, C)

        q_v = q_2d.permute(0, 2, 1, 3).reshape(B * W, H, C)
        k_v = k_2d.permute(0, 2, 1, 3).reshape(B * W, H, C)
        v_v = v_2d.permute(0, 2, 1, 3).reshape(B * W, H, C)
        Va_v, _ = self.mha_v(q_v, k_v, v_v)
        Va_v = Va_v.reshape(B, W, H, C).permute(0, 2, 1, 3)

        Va = (Va_h + Va_v).reshape(B, HW, C)

        qkv_2d    = torch.cat([q_2d, k_2d, v_2d], dim=3).permute(0, 3, 1, 2)
        Vu        = self.dwconv(qkv_2d)
        Vs_detail = self.pw_conv(Vu).flatten(2).permute(0, 2, 1)
        Vs        = Va * torch.sigmoid(Vs_detail)

        alpha_bar    = torch.einsum('bsc,nc->bns', Vs, self.w_local) / self.C
        alpha        = F.softmax(alpha_bar, dim=-1)
        Vprime_local = torch.bmm(alpha, Vs)

        return Vprime_local  # (B, N, C)


# ═══════════════════════════════════════════════════════════════════════════════
# 6c. Feature Interaction Decoder (FID) — Enhancement 2
#     Transformer Decoder (L=6) replaces the LSTM.
#
#     Architecture:
#       • Input projection: V′local (B, N, local_feat_dim) → (B, N, d_model)
#       • Query: learned position embeddings + word embeddings over t=0..T-1
#       • L=6 Transformer decoder layers (causal self-attn + cross-attn)
#       • Output projection: (B, T, d_model) → (B, T, embed_dim=300)
#       • Same smooth-L1 loss, same beam-search API as before
# ═══════════════════════════════════════════════════════════════════════════════

class FID(nn.Module):
    """
    Feature Interaction Decoder — Transformer Decoder variant.
    Replaces the LSTM decoder while keeping the same external API.
    """
    def __init__(self, cfg, emb_matrix):
        super().__init__()
        V   = cfg['vocab_size']
        D   = cfg['embed_dim']          # 300
        L   = cfg['local_feat_dim']     # 2048  (V′local feature dim)
        dm  = cfg['dec_d_model']        # 512
        nh  = cfg['dec_nhead']          # 8
        dff = cfg['dec_dim_feedfwd']    # 2048
        drop= cfg['dec_dropout']        # 0.1
        n_layers = cfg['dec_layers']    # 6
        max_len  = cfg['max_seq_len']   # 30
        self.cfg = cfg

        # Word embedding (GloVe-initialised, fine-tunable)
        self.embedding = nn.Embedding(V, D, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

        # Project word embedding D → d_model
        self.word_proj = nn.Linear(D, dm)

        # Learned positional embeddings for decoder queries
        self.pos_emb = nn.Embedding(max_len, dm)

        # Project Vglobal (D=300) to d_model for MLP M() equivalent
        self.global_proj = nn.Linear(D, dm)

        # Project V′local (local_feat_dim) → d_model for cross-attention
        self.mem_proj = nn.Linear(L, dm)

        # L=6 Transformer decoder layers
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dm, nhead=nh, dim_feedforward=dff,
            dropout=drop, batch_first=True, norm_first=True  # pre-norm for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_layers,
            norm=nn.LayerNorm(dm)
        )

        # Output projection d_model → embed_dim (D=300, GloVe space)
        self.out_proj = nn.Linear(dm, D)

        self.dm = dm
        self.D  = D
        

    def _make_causal_mask(self, T, device):
        """Upper-triangular causal mask (True = ignore)."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, Vprime_local, Vglobal, captions, lengths):
        B, N, _ = Vprime_local.shape
        T = captions.size(1) - 1
        device = Vprime_local.device
    
        # Memory: project V′local to d_model
        memory = self.mem_proj(Vprime_local)            # (B, N, dm)
    
        # Query construction
        tok_ids  = captions[:, :T]                      # (B, T)
        word_emb = self.word_proj(self.embedding(tok_ids))
        pos_idx  = torch.arange(T, device=device).unsqueeze(0)
        pos_emb  = self.pos_emb(pos_idx)
        g_bias   = self.global_proj(Vglobal).unsqueeze(1)
        tgt      = word_emb + pos_emb + g_bias          # (B, T, dm)
    
        # Causal mask
        causal_mask = self._make_causal_mask(T, device) # (T, T)
    
        # ── Padding mask with NaN guard ───────────────────────────────────────────
        pad_mask = (tok_ids == self.cfg.get('pad_id', 0))   # (B, T)  True=ignore
    
        # If ALL positions in a row are masked → attention softmax gets -inf
        # everywhere → NaN. Unmask those rows entirely to avoid this.
        all_pad_rows = pad_mask.all(dim=1)              # (B,)
        if all_pad_rows.any():
            pad_mask = pad_mask.clone()                 # don't modify in-place
            pad_mask[all_pad_rows] = False              # safe: these are dummy rows
    
        # Transformer decoder
        out = self.transformer_decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask
        )                                               # (B, T, dm)
    
        pred_embs = self.out_proj(out)                  # (B, T, D=300)
        return pred_embs

    @torch.no_grad()
    def decode_beam(self, Vprime_local, Vglobal, sos_id, eos_id, max_len, beam_size=3):
        device  = Vprime_local.device
        memory  = self.mem_proj(Vprime_local)        # (1, N, dm)
        g_bias  = self.global_proj(Vglobal).unsqueeze(1)  # (1, 1, dm)
    
        beams     = [(0.0, [sos_id])]
        completed = []
    
        min_len  = 4        # was 5 — allow slightly shorter captions
        hard_max = max_len
    
        for step in range(hard_max):
            if not beams:
                break
            all_candidates = []
    
            for score, tokens in beams:
                # ── If last token is EOS, beam is done ───────────────────────────
                if tokens[-1] == eos_id:
                    length_penalty = max(len(tokens), 1) ** 1.0  # was 0.7 → 1.0 penalises long more
                    completed.append((score / length_penalty, tokens))
                    continue
    
                T_cur      = len(tokens)
                tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
                # ── Build decoder input ───────────────────────────────────────────
                word_emb = self.word_proj(self.embedding(tok_tensor))
                pos_idx  = torch.arange(T_cur, device=device).unsqueeze(0)
                pos_emb  = self.pos_emb(pos_idx)
                tgt      = word_emb + pos_emb + g_bias
    
                # ── Transformer decoder forward ───────────────────────────────────
                causal_mask = self._make_causal_mask(T_cur, device)
                out         = self.transformer_decoder(tgt, memory, tgt_mask=causal_mask)
                last_out    = out[:, -1, :]
                out_emb     = self.out_proj(last_out)
    
                # ── Vocabulary logits via dot product ─────────────────────────────
                logits    = out_emb @ self.embedding.weight.T    # (1, V)
                log_probs = F.log_softmax(logits, dim=-1)[0]     # (V,)
    
                # ── Repetition penalty (recent 6 tokens) ─────────────────────────
                recent = set(tokens[-6:])
                for tid in recent:
                    if tid not in (sos_id, eos_id):
                        log_probs[tid] = log_probs[tid] - 5.0    # was 3.0 → 5.0
    
                # ── Trigram blocking ──────────────────────────────────────────────
                if len(tokens) >= 3:
                    for i in range(len(tokens) - 2):
                        if tuple(tokens[i:i+2]) == tuple(tokens[-2:]):
                            log_probs[tokens[i+2]] = log_probs[tokens[i+2]] - 10.0
    
                # ── Length control ────────────────────────────────────────────────
                if T_cur < min_len:
                    log_probs[eos_id] = log_probs[eos_id] - 1000.0
                if T_cur >= hard_max - 1:
                    log_probs[eos_id] = log_probs[eos_id] + 1000.0
    
                # ── Progressively encourage EOS as length grows ───────────────────
                # This is the key fix for ratio=1.97 — gently push toward ending
                if T_cur > 10:
                    log_probs[eos_id] = log_probs[eos_id] + 0.3 * (T_cur - 10)
    
                # ── Expand beam ───────────────────────────────────────────────────
                topk_scores, topk_ids = log_probs.topk(beam_size * 2)
                for s, tid in zip(topk_scores.tolist(), topk_ids.tolist()):
                    all_candidates.append((score + s, tokens + [tid]))
    
            if not all_candidates:
                break
    
            # ── Prune: keep top beam_size by length-normalised score ─────────────
            all_candidates.sort(
                key=lambda x: x[0] / (max(len(x[1]), 1) ** 1.0),  # was 0.7 → 1.0
                reverse=True
            )
            beams = all_candidates[:beam_size]
    
        # ── Collect any beams that never hit EOS ─────────────────────────────────
        for score, tokens in beams:
            length_penalty = max(len(tokens), 1) ** 1.0            # was 0.7 → 1.0
            completed.append((score / length_penalty, tokens))
    
        if not completed:
            return []
    
        # ── Pick best completed sequence ──────────────────────────────────────────
        completed.sort(key=lambda x: x[0], reverse=True)
        best = completed[0][1]
    
        # ── Strip SOS / EOS / PAD ─────────────────────────────────────────────────
        out = []
        for t in best:
            if t == sos_id:
                continue
            if t == eos_id:
                break
            out.append(t)
        return out
    
# ══════════════════════════════════════════════════════════════════════════════
# 6d.                            Full TSFE model 
# ═══════════════════════════════════════════════════════════════════════════════

class TSFE(nn.Module):
    def __init__(self, cfg, emb_matrix):
        super().__init__()
        # Enhancement 1: Swin Transformer V2 Base (256×256 window8)
        # out_indices=(1,2,3) → F2(256ch), F3(512ch), F4(1024ch)
        self.encoder = timm.create_model(
            'swinv2_base_window8_256',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3)
        )
        self.amff = AMFF(cfg)
        self.lfse = LFSE(cfg)
        self.fid  = FID(cfg, emb_matrix)  # Enhancement 2: Transformer decoder
        self.cfg  = cfg

    def encode(self, images):
        """Run encoder + AMFF + LFSE. Returns Vprime_local, Vglobal."""
        feats = self.encoder(images)   # [F2, F3, F4]
        f2, f3, f4 = feats[0], feats[1], feats[2]
        # SwinV2 outputs (B, H, W, C) — convert to (B, C, H, W)
        f2 = f2.permute(0, 3, 1, 2).contiguous()
        f3 = f3.permute(0, 3, 1, 2).contiguous()
        f4 = f4.permute(0, 3, 1, 2).contiguous()

        Vlocal, Vglobal, H, W = self.amff(f2, f3, f4)
        Vprime_local = self.lfse(Vlocal, H, W)
        return Vprime_local, Vglobal

    def forward(self, images, captions, lengths):
        Vprime_local, Vglobal = self.encode(images)
        pred_embs = self.fid(Vprime_local, Vglobal, captions, lengths)
        return pred_embs

    @torch.no_grad()
    def caption(self, image, max_len, beam_size=3):
        """Generate caption for a single image tensor (1, C, H, W)."""
        self.eval()
        Vprime_local, Vglobal = self.encode(image)
        sos = self.cfg.get('sos_id', 1)  # default SOS token id
        eos = self.cfg.get('eos_id', 2)  # default EOS token id
        ids = self.fid.decode_beam(Vprime_local, Vglobal, sos, eos, max_len, beam_size)
        return ids
