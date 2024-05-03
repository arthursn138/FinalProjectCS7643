import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import FlavaModel, FlavaFeatureExtractor, BertTokenizer
from engine.datasets import ucf101
from torch.utils.data import DataLoader
from engine.datasets import dataset_classes
from engine.tools.utils import makedirs, set_random_seed
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import torch
from PIL import Image


set_random_seed(1)

data_dir = "/home/gkaviani3/pythonProject/cross_modal_adaptation/data/"
# Create an instance of the UCF101 dataset
ucf101 = dataset_classes["ucf101"](data_dir)
# print(type(ucf101))
# ucf101 contains train, val, and test attributes with the data loaded

train_loader = DataLoader(ucf101.train, batch_size=32, shuffle=True)
val_loader = DataLoader(ucf101.val, batch_size=32, shuffle=False)
test_loader = DataLoader(ucf101.test, batch_size=32, shuffle=False)


print("Torch version:", torch.__version__)

flava = FlavaModel.from_pretrained("facebook/flava-full").eval().to("cuda")
fe = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
tokenizer = BertTokenizer.from_pretrained("facebook/flava-full")

input_resolution = flava.config.image_config.image_size
context_length = 77 # Used for FLAVA multimodal tasks
vocab_size = flava.config.text_config.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in flava.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)



# Get a list of all the files in the directory
files = os.listdir('/home/gkaviani3/pythonProject/cross_modal_adaptation/data/ucf101/UCF-101-midframes')

# Create a list of class names by removing the file extension from each file name
ucf101_classes = [os.path.splitext(file)[0] for file in files]
print(ucf101_classes , len(ucf101_classes))

ucf101_templates = [
    'a photo of a person doing {}.'
]

# # Path to your dataset
# data_dir = "/home/gkaviani3/pythonProject/cross_modal_adaptation/data/ucf101/UCF-101-midframes"
#
# # Load the dataset using ImageFolder structure
# dataset = load_dataset('imagefolder', data_dir=data_dir)
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

dataset_name = "ucf101"
# zeroshot_weights = zeroshot_classifier(ucf101_classes, ucf101_templates)
# torch.save(zeroshot_weights, f'/home/gkaviani3/pythonProject/cross_modal_adaptation/Flava/zeroshot_weights/zeroshot_weights_{dataset_name}.pt')
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
    plt.title('t-SNE of Class Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(fig_name)
    plt.show()

plot_text_embeddings(zeroshot_weights , ucf101_classes, "/home/gkaviani3/pythonProject/cross_modal_adaptation/Flava/figures/ucf101_class_embeddings.png")



def classify_image(image_path, flava_model, feature_extractor, zeroshot_weights, classnames):
    """
    Classify an image using zero-shot learning with the FLAVA model and pre-computed text embeddings.

    Args:
    image_path (str): Path to the image to classify.
    flava_model (FlavaModel): Pre-trained FLAVA model.
    feature_extractor (FlavaFeatureExtractor): FLAVA feature extractor.
    zeroshot_weights (torch.Tensor): Pre-computed zero-shot weights for class text embeddings.
    classnames (list): List of class names corresponding to zeroshot_weights.

    Returns:
    str: Predicted class name.
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

    # Get the predicted class index
    predicted_class_idx = torch.argmax(logits, dim=1)  # dim=1 because logits is now (1, 101)
    predicted_class = classnames[predicted_class_idx.item()]  # Convert tensor to integer

    return predicted_class


# Usage example:

image_path = ucf101.test[0]['impath']
predicted_class = classify_image(image_path, flava, fe, zeroshot_weights, ucf101_classes)
print(f"The image was classified as: {predicted_class}" , f"correct class name is {ucf101.test[0]['classname']}")
