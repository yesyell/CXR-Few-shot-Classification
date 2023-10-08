import os
from PIL import Image
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import clip
#Constructs a CLIP processor which wraps a CLIP image processor and a CLIP tokenizer into a single processor.
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import re
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from transformers import AdamW
import math
from dataset import CustomDataset

# Hyperparameters
batch_size = 128
learning_rate = 3e-6
num_epochs = 40 #10
temperature = 0.05

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5"
GPU_NUM = 2 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')

# Load the CLIP model with RN50 as the image encoder
model, preprocess = clip.load("RN50", device)


# Paths to train and validation CSV files
train_csv_file = './train_data_p10.csv'
val_csv_file = './val_data_p10.csv'

# Root directory where the images and text files are located
root_dir = '/nas/user/jaeryung/mimic-cxr' # Modify this to match the new path prefix

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

# Specify the number of samples you want to use for training and validation
train_sample_limit = 18888 #10000  # Number of samples for training
val_sample_limit = 4722 #3000    # Number of samples for validation

# Create separate datasets for train and validation
train_dataset = CustomDataset(csv_file=train_csv_file, root_dir=root_dir, transform=transform, sample_limit=train_sample_limit)
val_dataset = CustomDataset(csv_file=val_csv_file, root_dir=root_dir, transform=transform, sample_limit=val_sample_limit)

# Define train and validation data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Now you have separate train_dataset and val_dataset instances for training and validation.

def extract_findings(text):
    findings_match = re.search(r'(?:FINDINGS|IMPRESSIONS?):(.*?)(?:IMPRESSION:|$)', text, re.DOTALL | re.IGNORECASE)
    if findings_match:
        findings_text = findings_match.group(1).strip()
    else:
        findings_text = text.strip()
    return findings_text

# Initialize the CLIP processor (Hugging Face Transformers library)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Define the maximum context length (77 tokens)
max_context_length = 77


# Initialize the optimizer with the CLIP model parameters and learning rate
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)

# Define a linear warm-up function for the learning rate
def linear_warmup(current_step, warmup_steps):
    return min(1.0, current_step / warmup_steps)

# Create a learning rate scheduler with cosine annealing and linear warm-up
warmup_steps = len(train_loader)  # 1 epoch linear warm-up
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / (num_epochs * len(train_loader)))),)

# Define the CrossEntropyLoss for image and text losses
loss_image = nn.CrossEntropyLoss()
loss_text = nn.CrossEntropyLoss()

start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()  # Set the model to training mode

    # Use tqdm to display a progress bar
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for batch_images, batch_texts in train_loader:
            batch_images = batch_images.to(device)

            # Extract the section using regex for each text in the batch
            findings_texts = []
            for text in batch_texts:
              findings_text = extract_findings(text)
              findings_texts.append(findings_text)

            # Tokenize the text for the entire batch
            tokenized_findings = processor(text=findings_texts, return_tensors="pt", max_length=max_context_length, truncation=True, padding="max_length")
            token_ids_findings = tokenized_findings['input_ids']
            token_ids_findings = token_ids_findings.to(device)

            logits_per_image, logits_per_text = model(batch_images, token_ids_findings)
            ground_truth = torch.arange(len(batch_images), dtype=torch.long, device=device)

            loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_text, ground_truth)) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)  # Update the progress bar

        # Update the learning rate at the end of each epoch
        scheduler.step()

    end_time = time.time()

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_total_loss = 0.0
    with torch.no_grad():
        for batch_images, batch_texts in val_loader:
            # ... (same code as in the training loop)
            batch_images = batch_images.to(device)

            # Extract the section using regex for each text in the batch
            findings_texts = []
            for text in batch_texts:
              findings_text = extract_findings(text)
              findings_texts.append(findings_text)

            # Tokenize the "FINDINGS" text for the entire batch
            tokenized_findings = processor(text=findings_texts, return_tensors="pt", max_length=max_context_length, truncation=True, padding="max_length")
            token_ids_findings = tokenized_findings['input_ids']
            token_ids_findings = token_ids_findings.to(device)

            logits_per_image, logits_per_text = model(batch_images, token_ids_findings)
            ground_truth = torch.arange(len(batch_images), dtype=torch.long, device=device)

            loss = (loss_image(logits_per_image, ground_truth) + loss_text(logits_per_text, ground_truth)) / 2.0

            val_total_loss += loss.item()

    # Calculate and print the average loss for the epoch
    average_loss = total_loss / len(train_loader)
    val_average_loss = val_total_loss / len(val_loader)
    epoch_time = end_time - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {average_loss:.4f} Validation Loss: {val_average_loss:.4f} Time: {epoch_time:.2f} seconds")

    start_time = time.time()

# Training is complete
print("Training finished.")

torch.jit.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"./checkpoints/CLIP_model.pt") #just change to your preferred folder/filename
