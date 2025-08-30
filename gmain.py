import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import sys
import argparse
import numpy as np

# ======================================================================================
#
#   Model & Helper Definitions (Must match your final training pipeline)
#
# ======================================================================================

def mean_pool(last_hidden_state, attention_mask):
    """ Helper function for mean pooling. """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class TextEncoder(nn.Module):
    """
    The advanced TextEncoder with a projection head.
    This must match the definition used during contrastive training.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 proj_dim=384, freeze_backbone=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim)
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state
        pooled = mean_pool(last, attention_mask)
        z = self.proj(pooled)
        z = F.normalize(z, dim=-1)
        return z

class MLPClassifier(nn.Module):
    """
    The MLP classifier trained on top of the embeddings.
    This must match the definition used in the classification script.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.fc(x)


# ======================================================================================
#
#   Prediction Logic
#
# ======================================================================================

class Predictor:
    def __init__(self, query_encoder_path, pos_encoder_path, mlp_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models onto device: {self.device}")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        embedding_dim = 384 # For all-MiniLM-L6-v2

        # Load the three trained model components
        self.query_encoder = TextEncoder(model_name=model_name).to(self.device)
        self.pos_encoder = TextEncoder(model_name=model_name).to(self.device)
        self.mlp_classifier = MLPClassifier(input_dim=embedding_dim * 3).to(self.device)

        # Load the state dictionaries
        self.query_encoder.load_state_dict(torch.load(query_encoder_path, map_location=self.device))
        self.pos_encoder.load_state_dict(torch.load(pos_encoder_path, map_location=self.device))
        self.mlp_classifier.load_state_dict(torch.load(mlp_path, map_location=self.device))
        
        # Set models to evaluation mode
        self.query_encoder.eval()
        self.pos_encoder.eval()
        self.mlp_classifier.eval()
        print("All models loaded successfully.")

    def predict(self, hypothesis_text: str, target_text: str):
        """
        Performs a single prediction on a text pair.
        """
        # Tokenize inputs
        hyp_tokenized = self.tokenizer(hypothesis_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        tgt_tokenized = self.tokenizer(target_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        with torch.no_grad():
            # Generate embeddings using the two separate encoders
            hyp_emb = self.query_encoder(
                hyp_tokenized['input_ids'].to(self.device),
                hyp_tokenized['attention_mask'].to(self.device)
            )
            tgt_emb = self.pos_encoder(
                tgt_tokenized['input_ids'].to(self.device),
                tgt_tokenized['attention_mask'].to(self.device)
            )

            # Create the feature vector for the MLP
            feature_vector = torch.cat([hyp_emb, tgt_emb, torch.abs(hyp_emb - tgt_emb)], dim=1)

            # Get prediction from the MLP
            logits = self.mlp_classifier(feature_vector)
            probabilities = F.softmax(logits, dim=1)
            
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
            predicted_class_idx = predicted_class_idx.item()
            confidence = confidence.item()

        # Format and print the result
        label_map = {0: "Not Hallucination", 1: "Hallucination"}
        prediction_text = label_map[predicted_class_idx]

        print("\n--- Prediction Result ---")
        print(f"Hypothesis: '{hypothesis_text}'")
        print(f"Target:     '{target_text}'")
        print(f"\nPrediction: {prediction_text} (Confidence: {confidence:.2%})")
        print("-------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run hallucination detection using trained contrastive encoders and an MLP.")
    parser.add_argument("--hyp", type=str, required=True, help="The hypothesis text.")
    parser.add_argument("--tgt", type=str, required=True, help="The target/reference text.")
    parser.add_argument("--query_enc_path", type=str, default="query_encoder.pt", help="Path to the saved query encoder (.pt file).")
    parser.add_argument("--pos_enc_path", type=str, default="pos_encoder.pt", help="Path to the saved positive encoder (.pt file).")
    parser.add_argument("--mlp_path", type=str, default="mlp_classifier.pt", help="Path to the saved MLP classifier (.pt file).")
    args = parser.parse_args()

    # Check that all required model files exist
    for path in [args.query_enc_path, args.pos_enc_path, args.mlp_path]:
        if not os.path.exists(path):
            sys.exit(f"FATAL ERROR: Model file not found at '{path}'. Please run the training script first.")

    predictor = Predictor(
        query_encoder_path=args.query_enc_path,
        pos_encoder_path=args.pos_enc_path,
        mlp_path=args.mlp_path
    )
    predictor.predict(hypothesis_text=args.hyp, target_text=args.tgt)


if __name__ == "__main__":
    main()

