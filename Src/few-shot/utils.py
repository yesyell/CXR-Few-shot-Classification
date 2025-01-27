from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import clip

# Log-Sum-Exp-Sign loss function
def LSES(s, y, gamma=50, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
  
  s, y = s.detach().cpu().numpy(), y.detach().cpu().numpy()
  sum_exp = []
  for i in range(len(y)):
    sum_exp.append(np.exp(-y[i] * gamma * s[i]))
  logit = np.log(1 + np.sum(sum_exp))
  logit = torch.tensor(logit, requires_grad=True).to(device)

  return logit


class CustomImageDataset(Dataset):
  def __init__(self, annotations_file, img_dir, transform=None, sample_limit=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    # If a sample_limit is provided, limit the number of samples
    if sample_limit is not None:
        self.img_labels = self.img_labels.head(sample_limit)

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['path'])
    label = self.img_labels.iloc[idx]['class']
    #image = read_image(img_path)
    image = Image.open(img_path).convert("RGB")
    if self.transform:
      image = self.transform(image)
    return image, label
  

# normalized features
def normalize_features(features):
    features = features / features.norm(dim=1, keepdim=True)
    return features

# get text embeddings for structured texts
def get_text_emb(text_encoder, texts, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    zeroshot_weights = []
    for text in texts:
      text = clip.tokenize(text).to(device)
      class_embeddings = text_encoder.encode_text(text)
      class_embedding = normalize_features(class_embeddings)
      class_embedding = class_embeddings.mean(dim=0)
      class_embedding /= class_embedding.norm()
      zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

    return zeroshot_weights

def one_hot_encode(label):
    batch_size = label.shape[0]
    y = torch.tensor((), dtype=torch.int64)
    y = y.new_ones((batch_size, 98))
    y *= -1
    for b in range(batch_size):
        y[b][label] = 1
    return y

