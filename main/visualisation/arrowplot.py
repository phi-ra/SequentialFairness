import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from ..fairness.metrics import unfairness

def add_arrow(line, position=None, direction='right', size=30, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    
    try:
        if position is None:
            position = ydata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(ydata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color, lw = 1),
            size=size
        )
    except:
        if position is None:
            position = xdata.mean()
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color, lw = 2),
            size=size
        )


def viz_fairness_distrib(fair_models,
                         model,
                         X_test,
                         y_test,
                         a_indices = [1],
                         display_fig = True,
                         unf_index = 1,
                         titles=None):
    if display_fig:
        fig = plt.figure(figsize=(12, 7))
    n_a  = len(a_indices)
    n_m = len(fair_models)
    unfs = []
    perfs = []
    for i_a, a_index in enumerate(a_indices):
        # on test data
        y_pred = model.predict(X_test)
        y_pred0 = y_pred[X_test[:, -a_index] == -1]
        y_pred1 = y_pred[X_test[:, -a_index] == 1]

        perf = np.round(mean_squared_error(y_pred, y_test), 2)
        unf = np.round(unfairness(y_pred0, y_pred1), 2)
        
        if display_fig:
            plt.subplot(n_a, n_m + 1, i_a * (n_m+1) + 1)
            sns.kdeplot(y_pred0, fill= True, linewidth= 2, label = f"$A_{a_index}=1$", alpha = 0.2)
            sns.kdeplot(y_pred1, fill= True, linewidth= 2, label = f"$A_{a_index}=2$", alpha = 0.2)
            plt.title("Base model\n(error, $\mathcal{U}$"+f"$_{a_index}$) = ({perf}, {unf})", fontsize=11)
            plt.legend()
        if a_index == unf_index:
            unfs.append(unf)
            perfs.append(perf)
        for i, fair_model in enumerate(fair_models):
            add_title = ""
            # on test data
            try:
                y_pred = fair_model.predict(X_test) #y_pred_fair[a_index]
                y_pred0 = y_pred[X_test[:, -a_index] == -1] #fair_model.y_pred_fair0[a_index]#
                y_pred1 = y_pred[X_test[:, -a_index] == 1] #fair_model.y_pred_fair1[a_index]#
                perf = np.round(mean_squared_error(y_pred, y_test), 2)
                unf = np.round(unfairness(y_pred0, y_pred1), 2)
                if display_fig:
                    plt.subplot(n_a, n_m + 1, i * (n_m+1) + i_a + 2)
                    add_title = "\n(error, $\mathcal{U}$"+f"$_{a_index}$) = ({perf}, {unf})"
                    sns.kdeplot(y_pred0, fill= True, linewidth= 2, label = f"$A_{a_index}=1$", warn_singular=False, alpha = 0.2)
                    sns.kdeplot(y_pred1, fill= True, linewidth= 2, label = f"$A_{a_index}=2$", warn_singular=False, alpha = 0.2)
                    plt.title(titles[i] + add_title, fontsize=11)
                    plt.legend()
                if a_index == unf_index:
                    unfs.append(unf)
                    perfs.append(perf)
            except:
                pass
    if display_fig:
        fig.tight_layout()
        plt.show()
    return unfs, perfs

