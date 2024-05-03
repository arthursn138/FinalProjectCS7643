import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from transformers import FlavaModel, FlavaFeatureExtractor, BertTokenizer
from torch.utils.data import DataLoader
from engine.datasets import dataset_classes
from engine.datasets.benchmark import generate_fewshot_dataset
from engine.tools.utils import makedirs, set_random_seed
from engine.datasets.utils import DatasetWrapper, get_few_shot_setup_name, get_few_shot_benchmark
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns

set_random_seed(1)

data_dir = "/home/gkaviani3/pythonProject/cross_modal_adaptation/data/"
dataset_name = "oxford_flowers"
#Loading dataset
dataset_benchmark = dataset_classes[dataset_name](data_dir)
train_loader = DataLoader(dataset_benchmark.train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_benchmark.val, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset_benchmark.test, batch_size=32, shuffle=False)
#Pretrain model initialization
flava = FlavaModel.from_pretrained("facebook/flava-full").eval().to("cuda")
fe = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

#Define model configurations
input_resolution = flava.config.image_config.image_size
context_length = 77 # Used for FLAVA multimodal tasks
vocab_size = flava.config.text_config.vocab_size
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in flava.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)

#Create list of class names
files = os.listdir('/home/gkaviani3/pythonProject/cross_modal_adaptation/data/ucf101/UCF-101-midframes')

# Create a list of class names by removing the file extension from each file name
# dataset_benchmark_classes = [os.path.splitext(file)[0] for file in files]
dataset_benchmark_classes = dataset_benchmark.classnames
# print(ucf101_classes , len(ucf101_classes))
#Add text template for class names

dataset_benchmark_templates = [
    'a photo of a {}, a type of flower.'
]
# =[
#     'a photo of a {}.'
# ]
# dataset_benchmark_templates = [
#     'a photo of a person doing {}.'
# ]
#Extract Class name embeddings
def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts, return_tensors="pt", max_length=77, padding=True).to("cuda") #tokenize
            class_embeddings = flava.get_text_features(**texts)[:, 0, :] #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

#Save or load weights
zeroshot_weights = zeroshot_classifier(dataset_benchmark_classes, dataset_benchmark_templates)
torch.save(zeroshot_weights, f'/home/gkaviani3/pythonProject/cross_modal_adaptation/Flava/zeroshot_weights/zeroshot_weights_{dataset_name}.pt')
zeroshot_weights = torch.load(f'/home/gkaviani3/pythonProject/cross_modal_adaptation/Flava/zeroshot_weights/zeroshot_weights_{dataset_name}.pt')

def plot_text_embeddings(embeddings, class_labels , fig_name):
    # Assuming zeroshot_weights is available from zeroshot_classifier function
    embeddings = embeddings.cpu().detach().numpy().T  # Transpose to shape (num_classes, embedding_dim)

    # Reduce dimensions to 2 for visualization
    tsne = TSNE(n_components=2, random_state=1)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], palette="viridis")
    for i, txt in enumerate(class_labels):
        plt.annotate(txt, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    plt.title(f'Class Embeddings of {dataset_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(fig_name)
    plt.show()

plot_text_embeddings(zeroshot_weights , dataset_benchmark_classes, f"/home/gkaviani3/pythonProject/cross_modal_adaptation/Flava/figures/{dataset_name}_class_embeddings.png")

#Get logits of all classes for zeroshot test
def get_image_logits(image_path, flava_model, feature_extractor, zeroshot_weights):
    """
    Obtain logits for an image using zero-shot learning with the FLAVA model.

    Args:
    image_path (str): Path to the image to classify.
    flava_model (FlavaModel): Pre-trained FLAVA model.
    feature_extractor (FlavaFeatureExtractor): FLAVA feature extractor.
    zeroshot_weights (torch.Tensor): Pre-computed zero-shot weights for class text embeddings.

    Returns:
    torch.Tensor: Logits for all classes.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = feature_extractor(image, return_tensors="pt").to("cuda")

    # Get image features from FLAVA model
    with torch.no_grad():
        image_features = flava_model.get_image_features(**image_tensor).squeeze(0)
        image_features = image_features.mean(dim=0)  # Pooling (mean) across patches/tokens
        image_features /= image_features.norm()  # Normalize the pooled features

    # Compute the dot product between image features and class embeddings
    logits = torch.matmul(image_features.unsqueeze(0), zeroshot_weights)  # Add batch dimension to image_features

    return logits


#Evaluate Zero shot classification
def evaluate_zero_shot_classification(model, dataset, feature_extractor, zeroshot_weights, classnames):
    """
    Evaluate zero-shot classification model over a dataset to get Top-1 and Top-5 accuracies.

    Args:
    model (FlavaModel): A pre-trained FLAVA model.
    dataset (Dataset): A PyTorch Dataset containing the test images and labels.
    feature_extractor (FlavaFeatureExtractor): A FLAVA feature extractor.
    zeroshot_weights (torch.Tensor): The class embeddings tensor.
    classnames (list): The list of class names corresponding to class embeddings.

    Returns:
    float, float: Top-1 and Top-5 accuracies.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    top1, top5 = 0, 0
    total = 0

    for sample in loader:
        image_path, target = sample['impath'][0], sample['label']  # assuming the dataset returns a dictionary
        logits = get_image_logits(image_path, model, feature_extractor, zeroshot_weights)
        topk_vals, topk_indices = torch.topk(logits, 5)

        top1 += (topk_indices[:, 0] == target.to("cuda")).sum().item()
        top5 += (topk_indices == target.to("cuda").view(-1, 1)).any(dim=1).sum().item()
        total += 1

    top1_acc = top1 / total * 100
    top5_acc = top5 / total * 100

    return top1_acc, top5_acc


# Example usage:
# Assuming the dataset and other components are already initialized and configured
top1_accuracy, top5_accuracy = evaluate_zero_shot_classification(flava, dataset_benchmark.test, fe, zeroshot_weights, dataset_benchmark_classes)
print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
