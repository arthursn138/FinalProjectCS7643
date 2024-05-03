import torch
import torch.nn as nn
from engine.tools.utils import set_random_seed

import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, class_embedding_dim):
        super(MultimodalModel, self).__init__()
        self.fusion_layer = None
        self.class_embedding_dim = class_embedding_dim  # Store the dimension of class embeddings

    def forward(self, image_embeddings, text_embeddings):
        if self.fusion_layer is None:
            input_size = image_embeddings.shape[1] + text_embeddings.shape[1]
            # Initialize the fusion layer with the output size set to class embeddings dimension
            self.fusion_layer = nn.Linear(input_size, self.class_embedding_dim).to(image_embeddings.device)
            self.fusion_layer.reset_parameters()

        fused_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)
        fused_embeddings = self.fusion_layer(fused_embeddings)
        return fused_embeddings

# Usage example:
# model = MultimodalModel().to("cuda")
# fused_embeddings = model(image_embeddings, text_embeddings)