import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json

from transformers import (
    LlavaProcessor, LlavaForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq
)
from transformers import SmolVLMProcessor, SmolVLMForConditionalGeneration

def load_model_and_processor(model_id):
    if "llava" in model_id.lower():
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    elif "smolvlm" in model_id.lower():
        processor = SmolVLMProcessor.from_pretrained(model_id)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")
    
    return processor, model.eval()


class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        self.processor, self.model = load_model_and_processor(model_id)
        self.model_id = model_id
        self.meta_data = json.load(open("metadata_dict.json"))
    
    def forward(self, asin, image_path, prompt_template):
        if not os.path.exists(image_path):
            return None
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            return None

        prompt = prompt_template.format(
            item_title=self.meta_data[asin]['title'],
            item_desc=self.meta_data[asin]['description']
        )

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        inputs = {k: v.to(dtype=torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return pooled.squeeze(0).cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help="Model ID from Hugging Face")
    parser.add_argument('--image_dir', type=str, default="/home/spring2024/sk4858/HyperRec_old/data/All_Beauty")
    parser.add_argument('--output_file', type=str, default="item_embeddings.npy")
    parser.add_argument('--metadata_file', type=str, default="metadata_dict.json")
    args = parser.parse_args()

    # Update metadata file path globally
    if not os.path.exists(args.metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_file}")

    # Load ASINs
    train = pd.read_csv("data/All_Beauty.train.csv.gz")[['parent_asin']]
    val = pd.read_csv("data/All_Beauty.valid.csv.gz")[['parent_asin']]
    test = pd.read_csv("data/All_Beauty.test.csv.gz")[['parent_asin']]
    combined = pd.concat([train, val, test])
    item_encoder = LabelEncoder().fit(combined['parent_asin'])

    print(f"[INFO] Extracting embeddings for {len(item_encoder.classes_)} items using {args.model_id}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = EmbeddingExtractor(model_id=args.model_id, device=device)

    # You might need to update this depending on model output size
    embedding_dim = 4096
    embeddings = np.zeros((len(item_encoder.classes_), embedding_dim), dtype=np.float32)

    prompt_template = (
        "<image>Generate a short text description that captures the essence of Item titled [{item_title}] "
        "for use in a recommendation system. Item Description [ {item_desc} ]. "
        "Aim for a concise description that highlights key features, style, and use cases.\n?"
    )

    for idx, asin in enumerate(tqdm(item_encoder.classes_, desc="Extracting embeddings")):
        image_path = os.path.join(args.image_dir, f"{asin}_1.jpg")
        print(image_path)
        emb = extractor(asin, image_path, prompt_template)
        print(emb.shape)
        if emb is not None and emb.shape[0] == embedding_dim:
            embeddings[idx] = emb
        else:
            print(f"[WARN] Missing or incompatible embedding for {asin}, using zeros.")

    np.save(args.output_file, embeddings)
    print(f"[DONE] Saved embeddings to: {args.output_file}")

if __name__ == "__main__":
    main()
