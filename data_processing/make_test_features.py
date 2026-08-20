from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import open_clip


ROOT = Path("pilot_whiten_foveated")

CUBE_DATA_DIR = ROOT / "cube_ready" / "sub-01"
STIMULI_ROOT = ROOT / "stimuli"
FEATURE_DIR = ROOT / "features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu")

MODEL_NAME = "RN50"
PRETRAINED = "openai"

#flatten the CUBE img/text array and return unique string keys
def flatten_unique(arr):
    arr = np.asarray(arr).reshape(-1)

    return sorted(set(map(str, arr)))

#convert a stimulus key into its location in the stimuli directory.
def image_path_from_key(img_key):
    return STIMULI_ROOT / img_key.lstrip("/")

@torch.no_grad()
def encode_images(img_keys, model, preprocess, batch_size=8):

    model.eval()
    img_features = {}

    for i in tqdm(range(0, len(img_keys), batch_size), desc="Encoding test images"):

        batch_keys = img_keys[i:i + batch_size]

        images = []
        valid_keys = []

        for key in batch_keys:

            img_path = image_path_from_key(key)

            if not img_path.exists():
                raise FileNotFoundError(
                    f"Missing stimulus image: {img_path}")

            img = Image.open(img_path).convert("RGB")

            images.append(preprocess(img))

            valid_keys.append(key)

        image_tensor = torch.stack(images).to(DEVICE)

        feats = model.encode_image(image_tensor)

        feats = feats / feats.norm(dim=-1, keepdim=True)

        for key, feat in zip(valid_keys, feats):
            img_features[key] = (feat.detach().cpu().float())

        del image_tensor
        del feats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return img_features


@torch.no_grad()
def encode_texts(text_keys, model, batch_size=128):
    model.eval()

    device = next(model.parameters()).device

    text_features = {}

    for i in tqdm(range(0, len(text_keys), batch_size), desc="Encoding test texts"):

        batch_texts = text_keys[i:i + batch_size]

        tokens = open_clip.tokenize([f"This is a {text}." for text in batch_texts]).to(device)

        feats = model.encode_text(tokens)

        feats = feats / feats.norm(dim=-1, keepdim=True)

        for key, feat in zip(batch_texts, feats):
            text_features[key] = (feat.detach().cpu().float())

        del tokens
        del feats

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return text_features


def build_test_features():

    test_pt_path = (CUBE_DATA_DIR / "test.pt")

    if not test_pt_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_pt_path}")

    data = torch.load(test_pt_path, weights_only=False)

    img_keys = flatten_unique(data["img"])

    text_keys = flatten_unique(data["text"])

    print("\n========================================")
    print("TRUE TEST FEATURE EXTRACTION")
    print("========================================")

    print("Device:", DEVICE)
    print("Test file:", test_pt_path)
    print("Unique images:", len(img_keys))
    print("Unique texts:", len(text_keys))



    model, _, preprocess = (open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE,))


    img_features = encode_images(img_keys, model, preprocess)

    text_features = encode_texts(text_keys, model)



    out_path = (FEATURE_DIR / "test.pt")

    torch.save(
        {
            "img_features": img_features,
            "text_features": text_features,
        },
        out_path,
    )

    saved = torch.load(out_path, weights_only=False)

    print("\nSaved:", out_path)

    print("Saved image features:", len(saved["img_features"]))

    print("Saved text features:", len(saved["text_features"]))


    assert set(img_keys) == set(saved["img_features"].keys())

    assert set(text_keys) == set(saved["text_features"].keys())

    print("\nAll test feature checks passed.")


if __name__ == "__main__":
    build_test_features()