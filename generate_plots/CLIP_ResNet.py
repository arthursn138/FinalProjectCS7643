import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the data for caltech101
caltech101_data = {
    'shots': [1, 2, 4, 8, 16],
    'test_acc_mean_adapter': [0.890872211, 0.887897228, 0.910885734, 0.919269777, 0.930223124],
    'test_acc_std_adapter': [0.004228937, 0.020274057, 0.00432302, 0.001844244, 0.001194288],
    'test_acc_mean_linear': [0.888302907, 0.887221095, 0.910885734, 0.918593644, 0.928465179],
    'test_acc_std_linear': [0.001163262, 0.020696283, 0.003951762, 0.00257286, 0.00257286]
}

#
# # Plot with standard deviation as shaded area
# plt.figure(figsize=(10, 6))
# sns.set(style="whitegrid")
#
# # Adapter plot with shading for std deviation
# plt.plot(caltech101_data['shots'], caltech101_data['test_acc_mean_adapter'], label='Adapter', color='blue')
# plt.fill_between(caltech101_data['shots'],
#                  np.subtract(caltech101_data['test_acc_mean_adapter'], caltech101_data['test_acc_std_adapter']),
#                  np.add(caltech101_data['test_acc_mean_adapter'], caltech101_data['test_acc_std_adapter']),
#                  color='blue', alpha=0.2)
#
# # Linear plot with shading for std deviation
# plt.plot(caltech101_data['shots'], caltech101_data['test_acc_mean_linear'], label='Linear', color='orange')
# plt.fill_between(caltech101_data['shots'],
#                  np.subtract(caltech101_data['test_acc_mean_linear'], caltech101_data['test_acc_std_linear']),
#                  np.add(caltech101_data['test_acc_mean_linear'], caltech101_data['test_acc_std_linear']),
#                  color='orange', alpha=0.2)
#
# # Labels and Title
# plt.xlabel('Shots')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy for caltech101 Dataset')
# plt.legend()
#
# # Model and Backbone Text
# plt.text(0.8, 0.1, 'Model: "CLIP"\nBackbone: "ResNet50"', horizontalalignment='left', verticalalignment='bottom',
#          transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
#
# # Display plot with a tight layout
# plt.tight_layout()
# plt.show()

oxford_flowers_data = {
    'shots': [1, 2, 4, 8, 16],
    'test_acc_mean_adapter': [0.802408986, 0.859250237, 0.904723237, 0.933685208, 0.954933009],
    'test_acc_std_adapter': [0.005074661, 0.00552066, 0.005802125, 0.004476339, 0.00132602],
    'test_acc_mean_linear': [0.810393829, 0.862362972, 0.906888618, 0.940452023, 0.95926377],
    'test_acc_std_linear': [0.002016466, 0.006393835, 0.001566631, 0.006413856, 0.004197608]
}

ucf101_data = {
    'shots': [1, 2, 4, 8, 16],
    'test_acc_mean_adapter': [0.658031545, 0.702264517, 0.725526478, 0.764648868, 0.787205921],
    'test_acc_std_adapter': [0.007901771, 0.00872546, 0.000817131, 0.003145047, 0.002616837],
    'test_acc_mean_linear': [0.64225923, 0.698035069, 0.719094193, 0.754692043, 0.806105611],
    'test_acc_std_linear': [0.013620305, 0.006204341, 0.002267104, 0.003890986, 0.015168957]
}


# Plotting function
def plot_with_shading(data, title, ax):
    sns.set(style="whitegrid")

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
    ax.set_title(title)
    ax.legend()


# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))
plt.subplots_adjust(wspace=0.2)
# Plot for oxford_flowers
plot_with_shading(oxford_flowers_data, 'Test Accuracy for Oxford flowers Dataset', axes[0])

# Plot for ucf101
plot_with_shading(ucf101_data, 'Test Accuracy for Ucf101 Dataset', axes[1])

# Plot for Caltech101
plot_with_shading(caltech101_data, 'Test Accuracy for Caltech101 Dataset', axes[2])
# Model and Backbone Text for both subplots
for ax in axes:
    ax.text(0.65, 0.1, 'Model: "CLIP"\nBackbone: "ResNet50"', horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Display plot with a tight layout
plt.tight_layout()
plt.savefig('/home/gkaviani3/pythonProject/cross_modal_adaptation/generate_plots/CLIP_ResNet_linear_adapter.png')
plt.show()
