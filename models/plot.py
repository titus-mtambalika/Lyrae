import numpy as np
def confusion_matrix(ax, cm, classes = [1, 0], category = "value"):
	im = ax.imshow(cm, interpolation = "nearest", cmap = "Blues")
	ax.figure.colorbar(im, ax = ax)
	
	ax.set_xticks(np.arange(len(classes)))
	ax.set_yticks(np.arange(len(classes)))
	ax.set_yticklabels(classes)
	ax.set_xticklabels(classes)
	
	for i in range(len(classes)):
		for j in range(len(classes)):
			ax.text(j, i, cm[i, j], ha = "center", va = "center", color = "white" if cm[i, j] > cm.max() * 0.5 else "black")
			
	ax.set_xlabel(f"Predicted {category}")
	ax.set_ylabel(f"True {category}")

def loss_curve(ax, plt):
	ax.set_ylabel("Loss")
	ax.set_xlabel("Epochs")
	ax.legend()
	ax.grid()
	plt.tight_layout()