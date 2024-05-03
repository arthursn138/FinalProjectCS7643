import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


datasets_combined_data = {
    'caltech101': {
        'shots': [1, 2, 4, 8, 16],
        'test_acc_mean_RN50': [0.8891, 0.8892, 0.9153, 0.9251, 0.9378],
        'test_acc_std_RN50': [0.0070, 0.0187, 0.0034, 0.0035, 0.0019],
        'test_acc_mean_ViT-B16': [0.9405, 0.9417, 0.9577, 0.9609, 0.9647],
        'test_acc_std_ViT-B16': [0.0028, 0.0090, 0.0037, 0.0049, 0.0024]
    },
    'oxford_flowers': {
        'shots': [1, 2, 4, 8, 16],
        'test_acc_mean_RN50': [0.7633, 0.8277, 0.8904, 0.9329, 0.9593],
        'test_acc_std_RN50': [0.0038, 0.0040, 0.0040, 0.0019, 0.0026],
        'test_acc_mean_ViT-B16': [0.8429, 0.8997, 0.9448, 0.9681, 0.9793],
        'test_acc_std_ViT-B16': [0.0023, 0.0081, 0.0030, 0.0030, 0.0010]
    },
    'ucf101': {
        'shots': [1, 2, 4, 8, 16],
        'test_acc_mean_RN50': [0.6671, 0.7074, 0.7426, 0.7816, 0.8069],
        'test_acc_std_RN50': [0.0058, 0.0034, 0.0062, 0.0083, 0.0020],
        'test_acc_mean_ViT-B16': [0.7528, 0.7825, 0.8158, 0.8427, 0.8675],
        'test_acc_std_ViT-B16': [0.0050, 0.0063, 0.0051, 0.0028, 0.0013]
    }
}
def plot_with_shading(data, title, ax):
    sns.set(style="whitegrid")

    # Adapter plot with shading for std deviation
    ax.plot(data['shots'], data['test_acc_mean_RN50'], label='RN50', color='blue')
    ax.fill_between(data['shots'],
                    np.subtract(data['test_acc_mean_RN50'], data['test_acc_std_RN50']),
                    np.add(data['test_acc_mean_RN50'], data['test_acc_std_RN50']),
                    color='blue', alpha=0.2)

    # Linear plot with shading for std deviation
    ax.plot(data['shots'], data['test_acc_mean_ViT-B16'], label='ViT-B16', color='orange')
    ax.fill_between(data['shots'],
                    np.subtract(data['test_acc_mean_ViT-B16'], data['test_acc_std_ViT-B16']),
                    np.add(data['test_acc_mean_ViT-B16'], data['test_acc_std_ViT-B16']),
                    color='orange', alpha=0.2)

    # Labels and Title
    ax.set_xlabel('Shots')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(title)
    ax.legend()


# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7,13))
plt.subplots_adjust(wspace=0.2)
# Plot for oxford_flowers
plot_with_shading(datasets_combined_data["caltech101"], 'Test Accuracy for Oxford flowers Dataset', axes[0])

# Plot for ucf101
plot_with_shading(datasets_combined_data['oxford_flowers'], 'Test Accuracy for Ucf101 Dataset', axes[1])

# Plot for Caltech101
plot_with_shading(datasets_combined_data['ucf101'], 'Test Accuracy for Caltech101 Dataset', axes[2])
# Model and Backbone Text for both subplots
for ax in axes:
    ax.text(0.75, 0.1, 'Model: "CLIP"\nPartial Finetuning', horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Display plot with a tight layout
plt.tight_layout()
plt.savefig('/home/gkaviani3/pythonProject/cross_modal_adaptation/generate_plots/CLIP_Partial.png')
plt.show()
