import matplotlib.pyplot as plt
import numpy as np


def sparse_pyplot(adj, title=None, **plot_kw):
    """
    Create sparse Spy plot.
    """
    kw = {
        "precision": 0.1,
        "markersize": 5,
    }

    kw.update(plot_kw)

    fig = plt.figure(figsize=(7, 6))

    ax = fig.add_subplot(1, 1, 1)

    ax.spy(adj, **kw)

    _title = "Adjacency Spy Plot"

    title = _title if title is None else (_title + "\n" + title)

    ax.set_title(title, fontsize=14, fontweight="bold")

    return ax
