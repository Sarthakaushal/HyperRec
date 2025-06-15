import os
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# === Configuration ===
output_dir = "visualization"
os.makedirs(output_dir, exist_ok=True)

data = {
    ('LightGCN', ''): {'Recall': 0.0194, 'NDCG': 0.007},
    ('LGCN-SmolLM2.1-7B', 'General Text'): {'Recall': 0.028, 'NDCG': 0.011},
    ('LGCN-SmolLM2.1-7B', 'Chat Prompt'): {'Recall': 0.0414, 'NDCG': 0.0167},
    ('LGCN-LLaVA1.5-7B', 'General Text'): {'Recall': 0.0313, 'NDCG': 0.011},
    ('LGCN-LLaVA1.5-7B', 'Chat Prompt'): {'Recall': 0.032, 'NDCG': 0.012},
    ('LGCN-QWEN2.5', 'Chat Propmt'): {'Recall': 0.0384, 'NDCG': 0.014},
    ('LGCN-CLIP-ViT-large-patch3', 'General Text'): {'Recall': 0.0194, 'NDCG': 0.007},
}

categories = [f"{k[0]} - {k[1]}" for k in data.keys()]
recall_values = [d['Recall'] for d in data.values()]
ndcg_values = [d['NDCG'] for d in data.values()]
n_categories = len(categories)

# Angle for each axis
angles = [n / float(n_categories) * 2 * pi for n in range(n_categories)]
angles += angles[:1]  # Close the circle

# Define colors for each category text
category_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'cyan']
if len(category_colors) < n_categories:
    # Extend with a default color if needed
    category_colors.extend(['black'] * (n_categories - len(category_colors)))

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot Recall
values_recall = recall_values + recall_values[:1]
ax.plot(angles, values_recall, linewidth=2, linestyle='solid', label='Recall')
ax.fill(angles, values_recall, 'b', alpha=0.1)

# Plot NDCG
values_ndcg = ndcg_values + ndcg_values[:1]
ax.plot(angles, values_ndcg, linewidth=2, linestyle='solid', label='NDCG')
ax.fill(angles, values_ndcg, 'r', alpha=0.1)

# Set axis labels with different colors
ax.set_xticks(angles[:-1])
for i, label in enumerate(categories):
    ax.text(angles[i], ax.get_rmax() * 1.1, label, color=category_colors[i], ha='center', va='center')
ax.set_xticklabels([]) # Remove default xticklabels

# Set y-axis limits and labels
ax.set_yticks(np.linspace(0, max(max(recall_values), max(ndcg_values)) * 1.1, 5))
ax.set_yticklabels([f"{i:.3f}" for i in np.linspace(0, max(max(recall_values), max(ndcg_values)) * 1.1, 5)], fontsize=8)

# Title and legend
ax.set_title('Performance Comparison Accross Different VLLMs', size=14, color='black', y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Save the plot
save_path = os.path.join(output_dir, "colored_web_plot_performance.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Colored web plot saved to: {save_path}")