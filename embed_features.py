"""Shared image -> embedding module. THE no-skew keystone.

One frozen backbone (CLIP ViT-B/32), one preprocessing path, used by BOTH the
offline embed pipeline that fills the feature group and the online predictor that
scores an uploaded photo. Training and serving cannot diverge because they call the
same function on the same weights.

The embedding is L2-normalized (CLIP convention), 512-d. The backbone is frozen
everywhere: we never fine-tune, we only train heads on top, which is what makes the
whole system CPU-viable.
"""
import numpy as np
import torch

MODEL_ID = "openai/clip-vit-base-patch32"
EMBED_DIM = 512

_model = None
_proc = None


def _load():
    global _model, _proc
    if _model is None:
        from transformers import CLIPModel, AutoImageProcessor
        _model = CLIPModel.from_pretrained(MODEL_ID)
        _model.eval()
        _proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    return _model, _proc


def embed_images(pil_images, batch_size=64):
    """List of PIL images -> (n, 512) float32 L2-normalized numpy array."""
    model, proc = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inp = proc(images=batch, return_tensors="pt")
            f = model.get_image_features(**inp)
            if not isinstance(f, torch.Tensor):
                f = f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def zero_shot_country_scores(pil_images, country_names, batch_size=64):
    """CLIP zero-shot baseline: cosine of each image against 'a photo taken in X'.

    This is the honest baseline the trained head must beat: it costs nothing to
    'train' and already knows what countries look like.
    """
    model, proc = _load()
    from transformers import CLIPTokenizerFast
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    prompts = [f"a photo taken in {c}" for c in country_names]
    with torch.no_grad():
        t = tok(prompts, padding=True, return_tensors="pt")
        tf = model.get_text_features(**t)
        if not isinstance(tf, torch.Tensor):
            tf = tf.pooler_output
        tf = tf / tf.norm(dim=-1, keepdim=True)
    img = torch.from_numpy(embed_images(pil_images, batch_size))
    return (img @ tf.T).numpy()          # (n_images, n_countries) cosine scores
