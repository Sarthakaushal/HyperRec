import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
from src.Enums import FilePath
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    CLIPProcessor, CLIPModel
)
from transformers import SmolVLMProcessor, SmolVLMForConditionalGeneration

def load_model_and_processor(model_id ,quant = 8):
    if quant ==8:
        if "llava" in model_id.lower():
            processor = AutoProcessor.from_pretrained(model_id)
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
        elif "qwen" in model_id.lower():
            processor = AutoProcessor.from_pretrained(model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        elif "clip" in model_id.lower():
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda:0' if torch.cuda.device_count() else 'cpu')
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")
        return processor, model.eval()
    elif quant == 4:
        if "llava" in model_id.lower():
            processor = AutoProcessor.from_pretrained(model_id)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        elif "smolvlm" in model_id.lower():
            processor = SmolVLMProcessor.from_pretrained(model_id)
            model = SmolVLMForConditionalGeneration.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        elif "qwen" in model_id.lower():
            processor = AutoProcessor.from_pretrained(model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        elif "clip" in model_id.lower():
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda:0' if torch.cuda.device_count() else 'cpu')
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")
        return processor, model.eval()
    else:
        raise ValueError("Quanitzation can be either 8 or 4 bits")


class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, model_id, device, metadata = "metadata_dict.json", quant=8, image_dir = "data/All_Books"):
        super().__init__()
        self.processor, self.model = load_model_and_processor(model_id, quant=8)
        self.model.eval()
        self.model_id = model_id
        self.meta_data = json.load(open(metadata))
        self.decode_eng_text = False
        self.image_dir = image_dir
        
    def download_image(self, url, save_path):
        import requests
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Successfully downloaded image to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download image from {url}: {e}")
            return None
        return save_path
    
    def forward(self, asin, image_path, prompt_template):
        if not os.path.exists(image_path):
            print(f"[ERROR] Image 404 : {image_path}")
            try:
                image_path = self.download_image(self.meta_data[asin]['images'][0], 
                                                 self.image_dir+"/"+asin+"_1.jpg")
                print(image_path)
            except Exception as e:
                print(f"[ERROR] Failed to download image for {asin}: {e}")
                return None
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"[INFO] Successfully loaded image {image_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            return None

        prompt_template = prompt_template.format(
            item_title=self.meta_data[asin]['title'],
            item_desc=self.meta_data[asin]['description']
        )
        if self._get_model_id(self.model_id)== "clip":
            return self._get_clip_features(image_path, prompt_template)
            # image = Image.open(image_path).convert("RGB")
            # inputs = self.processor(
            #     text=prompt_template,
            #     images=image,
            #     return_tensors='pt',
            #     padding=True
            # ).to(self.model.device)
            # with torch.no_grad():
            #     outputs = self.model(**inputs)
            #     img_feat  = outputs.image_embeds   # (1, D)
            #     txt_feat  = outputs.text_embeds    # (1, D)
            # # combine into a single vector:
            # clip_feat = torch.cat([img_feat, txt_feat], dim=-1)
            # print(clip_feat.shape)
            # return clip_feat
        else: # LLM based procesing
            convo = self._get_conversation(self.model_id, prompt_template)
            
            prompt = self.processor.apply_chat_template(
                convo, add_generation_prompt=True)
            
            # raw_image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
            inputs = {k: v.to(dtype=torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                if self.decode_eng_text:
                    if self._get_model_id(self.model_id) == 'llava':
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=1200,      # Set an appropriate max length
                            num_beams=3,        # You can use beam search or sampling instead of greedy decoding
                            do_sample=False     # Set True for sampling, False for greedy or beam search
                        )
                        decoded_texts = self.processor.batch_decode(
                            generated_ids, skip_special_tokens=True
                        )
                        print(decoded_texts)
                    elif self._get_model_id(self.model_id) == 'smolvlm':
                        generated_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=64)
                        generated_texts = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                        )
                        print(generated_texts[0])
                    elif self._get_model_id(self.model_id) == 'qwen':
                        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                        ]
                        output_text = self.processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        print(output_text)
                    else:
                        print(f'Printing decodec model output not implemented: {self.model_id}')
                        pass
                hidden = outputs.hidden_states[-1]
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                return pooled.squeeze(0).cpu().numpy()
    def _get_clip_features(self, image_path: str, prompt_template: str, 
                  max_len: int = 77, stride: int = 64):
        image = Image.open(image_path).convert("RGB")
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            img_feat = self.model.get_image_features(**img_inputs)  # (1, D)

        # --- 2. Tokenize the full prompt and build overlapping spans ---
        tok = self.processor.tokenizer(
            prompt_template,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        spans: list[list[int]] = []
        for i in range(0, len(tok), stride):
            span_ids = tok[i : i + max_len]
            # # add special tokens (CLS/SEP) around each span
            # print(len(span_ids))
            # span_ids = (
            #     span_ids
            # )
            # print(len(span_ids))
            # print(span_ids)
            
            spans.append(span_ids)

        # --- 3. Encode each span (with the image) and collect text embeddings ---
        txt_feats = []
        for span_ids in spans:
            inputs = {
                "input_ids": torch.tensor([span_ids], device=self.model.device),
                "pixel_values": img_inputs.pixel_values,
                "attention_mask": torch.tensor([[1] * len(span_ids)], device=self.model.device),
            }
            with torch.no_grad():
                out = self.model(**inputs, return_dict=True)
                # or: out = self.model.get_text_features(**{"input_ids": inputs["input_ids"]})
                txt_feats.append(out.text_embeds)  # (1, D)

        # --- 4. Mean‑pool across all the span embeddings ---
        txt_feats = torch.cat(txt_feats, dim=0)         # (num_spans, D)
        txt_feat  = txt_feats.mean(dim=0, keepdim=True) # (1, D)

        # --- 5. Fuse and return ---
        clip_feat = torch.cat([img_feat, txt_feat], dim=-1)  # (1, 2D)
        return clip_feat
    
    def _get_conversation(self, model, prompt_text):
        model2chat_template = {
            "llava":[
                        {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                            ],
                        },
                    ],
            "smolvlm":[
                {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                            ],
                        },
            ],
            "qwen":[
                {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                            ],
                        },
            ]
        }
        return model2chat_template[self._get_model_id(model)]
    
    def _get_model_id(self, hf_id):
        if "llava" in hf_id.lower():
            return 'llava'
        elif "smolvlm" in hf_id.lower():
            return 'smolvlm'
        elif "qwen" in hf_id.lower():
            return "qwen"
        elif "clip" in hf_id.lower():
            return "clip"
        else:
            raise ValueError(f"Unsupported model_id: {hf_id}")
    
model2embedSize = {
    'llava':4096, # tested ok
    'smolvlm':2048, # TODO : Test 
    'qwen':3584,
    'clip': 1024
}

def get_model_id(hf_id):
    if "llava" in hf_id.lower():
       return 'llava'
    elif "smolvlm" in hf_id.lower():
        return 'smolvlm'
    elif "qwen" in hf_id.lower():
        return "qwen"
    elif "clip" in hf_id.lower():
            return "clip"
    else:
        raise ValueError(f"Unsupported model_id: {hf_id}")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help="Model ID from Hugging Face")
    parser.add_argument('--image_dir', type=str, default="/home/spring2024/sk4858/HyperRec_old/data/All_Beauty")
    parser.add_argument('--data_name', type=str, default="All_Beauty")
    parser.add_argument('--output_file', type=str, default="item_embeddings.npy")
    parser.add_argument('--metadata_file', type=str, default="metadata_dict.json")
    parser.add_argument('--quant', type=int, default=8)
    args = parser.parse_args()
    
    model_id = get_model_id(args.model_id)
    # Update metadata file path globally
    if not os.path.exists(args.metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_file}")
    root_data_dir = FilePath.ROOT_DATA_DIR.value
    # Load ASINs
    train = pd.read_csv(root_data_dir+f"data/{args.data_name}.train.csv.gz")[['parent_asin']]
    val = pd.read_csv(root_data_dir+f"data/{args.data_name}.valid.csv.gz")[['parent_asin']]
    test = pd.read_csv(root_data_dir+f"data/{args.data_name}.test.csv.gz")[['parent_asin']]
    combined = pd.concat([train, val, test])
    item_encoder = LabelEncoder().fit(combined['parent_asin'])

    print(f"[INFO] Extracting embeddings for {len(item_encoder.classes_)} items using {args.model_id}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = EmbeddingExtractor(model_id=args.model_id, device=device, 
                                   quant=args.quant, metadata=args.metadata_file, 
                                   image_dir = args.image_dir)

    # You might need to update this depending on model output size
    embedding_dim = model2embedSize[get_model_id(model_id)]
    embeddings = np.zeros((len(item_encoder.classes_), embedding_dim), dtype=np.float32)
    
    # prompt_template = extractor.processor.apply_chat_template(
    #     get_prompt_template(model_id, image_path, prompt_template),
    # )
    
    generation_prompt = (
        "Generate a short text description that captures the essence of Item titled [{item_title}] ",
        "for use in a recommendation system. Item Description [ {item_desc} ]. ",
        "Aim for a concise description that highlights key features, style, and use cases.\n?"
    )
    img_404 = []
    for idx, asin in enumerate(tqdm(item_encoder.classes_, desc="Extracting embeddings")):
        image_path = os.path.join(args.image_dir, f"{asin}_1.jpg")
        print(image_path)
        emb = extractor(asin, image_path,"".join(generation_prompt))
        if type(emb) == type(None):
            img_404.append(asin)
        else:
            emb_sample= emb
#  print(emb.shape)
        if emb is not None and emb.shape[0] == embedding_dim:
            embeddings[idx] = emb
        else:
            # embedding_dim[idx] = np.zeros_like(emb_sample)
            print(f"[WARN] Missing or incompatible embedding for {asin}, using zeros.")

    np.save(args.output_file, embeddings)
    print(f"[DONE] Saved embeddings to: {args.output_file}")

if __name__ == "__main__":
    main()