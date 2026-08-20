from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CUBE_DATA_DIR = Path("pilot2_whiten_foveated/cube_ready/sub-01")
STIMULI_ROOT = Path("pilot2_whiten_foveated/stimuli")
FEATURE_DIR = Path("pilot2_whiten_foveated/features")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "RN50"
PRETRAINED = "openai"


def flatten_unique(arr):
    arr = np.array(arr).reshape(-1)
    return sorted(set(map(str, arr)))


def image_path_from_key(img_key):
    return STIMULI_ROOT / img_key.lstrip("/")


@torch.no_grad()
def encode_images(img_keys, model, preprocess, batch_size=8):
    model.eval()

    img_features = {}

    for i in tqdm(range(0, len(img_keys), batch_size), desc="Encoding images"):
        batch_keys = img_keys[i:i + batch_size]

        images = []
        valid_keys = []

        for key in batch_keys:
            img_path = image_path_from_key(key)

            if not img_path.exists():
                raise FileNotFoundError(f"Missing stimulus image: {img_path}")

            img = Image.open(img_path).convert("RGB")
            images.append(preprocess(img))
            valid_keys.append(key)

        image_tensor = torch.stack(images).to(DEVICE)

        feats = model.encode_image(image_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        for key, feat in zip(valid_keys, feats):
            img_features[key] = feat.detach().cpu().float()

        del image_tensor
        del feats
        torch.cuda.empty_cache()

    return img_features


@torch.no_grad()
def encode_texts(text_keys, model, batch_size=128):
    device = next(model.parameters()).device
    text_features = {}

    for i in tqdm(range(0, len(text_keys), batch_size), desc="Encoding texts"):
        batch_texts = text_keys[i:i + batch_size]

        tokens = open_clip.tokenize(
            [f"This is a {t}." for t in batch_texts]
        ).to(device)

        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        for key, feat in zip(batch_texts, feats):
            text_features[key] = feat.float().cpu()

        del tokens, feats
        torch.cuda.empty_cache()

    return text_features


def build_features_for_split(split_name):
    pt_path = CUBE_DATA_DIR / f"{split_name}.pt"
    data = torch.load(pt_path, weights_only=False)

    img_keys = flatten_unique(data["img"])
    text_keys = flatten_unique(data["text"])

    print(f"\nSplit: {split_name}")
    print("Unique images:", len(img_keys))
    print("Unique texts:", len(text_keys))

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )

    img_features = encode_images(img_keys, model, preprocess)
    text_features = encode_texts(text_keys, model)

    out_path = FEATURE_DIR / f"{split_name}.pt"

    torch.save(
        {
            "img_features": img_features,
            "text_features": text_features,
        },
        out_path,
    )

    print("Saved:", out_path)


def main():
    build_features_for_split("train")
    build_features_for_split("test")


if __name__ == "__main__":
    main()