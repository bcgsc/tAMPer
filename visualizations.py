from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-colorblind')

def plot_PCA(X, y):
    """PCA plot vizualisation to 2 dimensions, assumes toxic(1) and non-toxic(0) data."""
    blue = y == 0
    green = y == 1
    pca = PCA(n_components=2, random_state=42)
    fig, ax = plt.subplots()
    pca.fit(X)
    transformed_X = pca.transform(X)
    ax.set_title("PCA on set (n={})".format(len(y)))
    ax.scatter(transformed_X[blue, 0], transformed_X[blue, 1],  label="non-toxic", alpha=0.8)
    ax.scatter(transformed_X[green, 0], transformed_X[green, 1], label="toxic", alpha=0.8)
    ax.legend(loc="upper right")
    ax.set_xlabel("Dimension 1 ({:.1f}% variance)".format(pca.explained_variance_ratio_[0]*100))
    ax.set_ylabel("Dimension 2 ({:.1f}% variance".format(pca.explained_variance_ratio_[1]*100))
    plt.show()
    return