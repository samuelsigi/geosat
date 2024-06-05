import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the confusion matrix values
TP = 92
TN = 80
FP = 8
FN = 0

# Create the confusion matrix
conf_matrix = np.array([[TP, FP],
                        [FN, TN]])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (92% Accuracy)')
plt.show()
