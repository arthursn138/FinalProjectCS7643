import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data for each dataset
datasets_data = {
    'oxford_flowers': {
        'shots': [1, 2, 4, 8, 16],
        'test_acc_mean_adapter': [0.8716, 0.9216, 0.9475, 0.9658, 0.9755],
        'test_acc_std_adapter': [0.0110, 0.0069, 0.0041, 0.0060, 0.0035],
        'test_acc_mean_linear': [0.8733, 0.9169, 0.9434, 0.9648, 0.9769],
        'test_acc_std_linear': [0.0073, 0.0051, 0.0022, 0.0048, 0.0038]
    },
    'ucf101': {
        'shots': [1, 2, 4, 8, 16],
        'test_acc_mean_adapter': [0.7440, 0.7721, 0.8033, 0.8271, 0.8453],
        'test_acc_std_adapter': [0.0155, 0.0139, 0.0041, 0.0082, 0.0061],
        'test_acc_mean_linear': [0.7350, 0.7713, 0.8004, 0.8164, 0.8351],
        'test_acc_std_linear': [0.0077, 0.0135, 0.0057, 0.0074, 0.0100]
    },
'caltech101': {
        'shots': [1, 2, 4, 8, 16],
        'test_acc_mean_adapter': [0.9360, 0.9364, 0.9502, 0.9548, 0.9581],
        'test_acc_std_adapter': [0.0016, 0.0045, 0.0017, 0.0034, 0.0031],
        'test_acc_mean_linear': [0.9333, 0.9373, 0.9502, 0.9513, 0.9571],
        'test_acc_std_linear': [0.0034, 0.0064, 0.0027, 0.0032, 0.0043]
    }
}

# Titles for each subplot
titles = {
    'caltech101': 'Test Accuracy for Caltech101 Dataset',
    'oxford_flowers': 'Test Accuracy for Oxford flowers Dataset',
    'ucf101': 'Test Accuracy for UCF101 Dataset'
}


# Function for plotting with shaded standard deviation
def plot_with_shading(data_dict, title_dict, backbone, nrows=3, ncols=1, figsize=(6, 10), wspace=0.2):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    plt.subplots_adjust(wspace=wspace)  # This adds space between each subplot

    for i, (dataset_name, data) in enumerate(data_dict.items()):
        ax = axes[i] if nrows > 1 else axes

        # Adapter plot with shading for std deviation
        ax.plot(data['shots'], data['test_acc_mean_adapter'], label='Adapter', color='blue')
        ax.fill_between(data['shots'],
                        np.subtract(data['test_acc_mean_adapter'], data['test_acc_std_adapter']),
                        np.add(data['test_acc_mean_adapter'], data['test_acc_std_adapter']),
                        color='blue', alpha=0.2)

        # Linear plot with shading for std deviation
        ax.plot(data['shots'], data['test_acc_mean_linear'], label='Linear', color='orange')
        ax.fill_between(data['shots'],
                        np.subtract(data['test_acc_mean_linear'], data['test_acc_std_linear']),
                        np.add(data['test_acc_mean_linear'], data['test_acc_std_linear']),
                        color='orange', alpha=0.2)

        # Labels and Title
        ax.set_xlabel('Shots')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(title_dict[dataset_name])
        ax.legend()

        # Model and Backbone Text
        ax.text(0.65, 0.1, f'Model: "CLIP"\nBackbone: "{backbone}"', horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Display plot with a tight layout
    plt.tight_layout()
    plt.savefig('/home/gkaviani3/pythonProject/cross_modal_adaptation/generate_plots/CLIP_ViT_linear_adapter.png')
    plt.show()


# Execute the plotting function with the provided data and settings
plot_with_shading(datasets_data, titles, 'ViT-B/16')
