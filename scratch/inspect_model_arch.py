import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import sys

sys.stdout.reconfigure(encoding='utf-8')

model_name = "indobenchmark/indobert-base-p2"
num_labels = 2
dropout_prob = 0.3
freeze_layers = 6

print(f"Loading model: {model_name}...")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.config.classifier_dropout = dropout_prob

print("\n=== MODEL CONFIGURATION ===")
config = model.config
print(f"Model Type             : {config.model_type}")
print(f"Vocabulary Size        : {config.vocab_size}")
print(f"Hidden Size (d_model)  : {config.hidden_size}")
print(f"Number of Layers (L)   : {config.num_hidden_layers}")
print(f"Attention Heads (A)    : {config.num_attention_heads}")
print(f"Intermediate Size (d_ff): {config.intermediate_size}")
print(f"Max Position Embeddings: {config.max_position_embeddings}")
print(f"Hidden Activation      : {config.hidden_act}")
print(f"Classifier Dropout Rate: {config.classifier_dropout}")

# Calculate parameters
total_params = sum(p.numel() for p in model.parameters())

# Embeddings params
emb_params = sum(p.numel() for p in model.bert.embeddings.parameters())

# Single encoder layer params
one_layer_params = sum(p.numel() for p in model.bert.encoder.layer[0].parameters())

# Pooler params
pooler_params = sum(p.numel() for p in model.bert.pooler.parameters())

# Classifier params
classifier_params = sum(p.numel() for p in model.classifier.parameters())

print("\n=== PARAMETER QUANTITIES ===")
print(f"Total Parameters         : {total_params:,}")
print(f"Embeddings Parameters    : {emb_params:,}")
print(f"One Encoder Layer Params : {one_layer_params:,}")
print(f"Pooler Parameters        : {pooler_params:,}")
print(f"Classifier Head Params   : {classifier_params:,}")

# Trainable vs Frozen
# Freeze embeddings and bottom N layers
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer_idx in range(freeze_layers):
    for param in model.bert.encoder.layer[layer_idx].parameters():
        param.requires_grad = False

frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nFrozen Parameters        : {frozen_params:,} (embeddings + {freeze_layers} encoder layers)")
print(f"Trainable Parameters     : {trainable_params:,} ({12 - freeze_layers} encoder layers + pooler + classifier)")
