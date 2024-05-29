import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
# np.random.seed(42)
x = np.random.rand(40)
x = np.concatenate((x, np.random.randn(15) * 0.2 + 0.4))
y = 3.5 * x + np.random.randn(55) * 1.65

# Create the plot
plt.figure(figsize=(8, 6))
sns.regplot(x=x, y=y, color='darkgreen', scatter_kws={'s': 150}, line_kws={'linewidth': 7})  # Increase line width

# Save the plot with a transparent background
plt.savefig('./FigureTable/manuscript_figures/regplot_transparent.png', transparent=True)
plt.show()
