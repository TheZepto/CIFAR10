from matplotlib import pyplot as plt
import numpy as np

# Plot the confusion matrix
# Updated function from the scikit-learn confusion matrix example
def save_confusion_matrix(cm, classes, index,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure() #Generate a new plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=1, vmax=1000)
    index += 1
    plt.title('Confusion matrix - epoch %d' % index)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    filename = 'saved_figures/figure_%03d.png' % index
    plt.savefig(filename, format= 'png')

    plt.close()