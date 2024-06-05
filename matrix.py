import numpy as np
import matplotlib.pyplot as plt

# Define the confusion matrix with 95% accuracy
conf_matrix = np.array([[95, 5, 0],
                        [5, 95, 0],
                        [0, 0, 100]])

# Plot confusion matrix
plt.figure(figsize=(8, 6), num='Confusion Matrix')  # Set the window title here
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.xlabel('Predicted value')
plt.ylabel('True label')
plt.title('Confusion Matrix (95% Accuracy)')
plt.xticks(range(conf_matrix.shape[1]), ['Class {}'.format(i) for i in range(conf_matrix.shape[1])])
plt.yticks(range(conf_matrix.shape[0]), ['Class {}'.format(i) for i in range(conf_matrix.shape[0])])
plt.tight_layout()
plt.show()
