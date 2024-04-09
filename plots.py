import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from PIL import Image
import gzip
import pickle as cp
import glob



def single_accuracy_plot(accs, methods,title='SplitMNIST',filename=None):
    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    plt.plot(np.arange(len(accs)) + 1, accs, marker='o')

    ax.set_xticks(list(range(1, len(accs) + 1)))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('Task')
    ax.set_title(title)
    ax.legend(labels=methods, title='Method')

    if filename is not None:
        fig.savefig("plots/{}.png".format(filename), bbox_inches='tight')



def split_three_mean_plot(accs1, accs2, meths1, meths2, filename=None):
    x = np.arange(1,6)

    data = accs1 + accs2
    methods = meths1 + meths2

    cmap = colourmap(methods) #tolist bc theyy are packed as np arrays
    mmap = markermap(methods)
    lmap = linestylemap(methods)

    mpl.rcParams['lines.markersize'] = 5
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['lines.marker'] = "s"
    mpl.rcParams['font.size'] = 12

    plt.clf()
    g, ax = plt.subplots(1, 3, figsize=(10,3), sharex=True,sharey=False)

    for i, acc in enumerate(data):
        ax[0].plot(x, np.nanmean(acc,0), color=cmap[i], linestyle=lmap[i], marker=mmap[i])
        if i<4:
            ax[1].plot(x, np.nanmean(acc,0), color=cmap[i], linestyle=lmap[i], marker=mmap[i])
        else:
            ax[2].plot(x, np.nanmean(acc,0), color=cmap[i], linestyle=lmap[i], marker=mmap[i])

    for i in range(3):
        ax[i].set_xticks(list(range(1, 6)))
        ax[i].set_xlabel('Task')
    ax[0].set_title("All")
    ax[1].set_title("VCL methods")
    ax[2].set_title("Coreset Only methods")

    ax[0].set_ylabel('Average accuracy')

    #ax[1].set_ylim([0.7,0.87])

    g.legend(   # The line objects
           labels=methods,
           loc="upper left",
            bbox_to_anchor=(1,1),   # Position of legend
           borderaxespad=0.3,    # Small spacing around legend box
           title="Methods:"  # Title for the legend
           )
    plt.tight_layout()

def split_mean_plot(data, methods, lower_bound=None):
    x = np.arange(1,6)

    cmap = colourmap(methods) #tolist bc theyy are packed as np arrays
    mmap = markermap(methods)
    lmap = linestylemap(methods)

    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['lines.marker'] = "s"

    fig = plt.figure(figsize=(7, 4), dpi=300)
    ax = plt.gca()

    for i, acc in enumerate(data):
        ax.plot(x, np.nanmean(acc,0), color=cmap[i], linestyle=lmap[i], marker=mmap[i])

    ax.set_xticks(list(range(1, 6)))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('Task')

    if lower_bound is not None:
        plt.ylim(lower_bound, 1.01)


    ax.legend(   # The line objects
           labels=methods,
           loc="best",   # Position of legend
           borderaxespad=0.3,    # Small spacing around legend box
           title="Methods:"  # Title for the legend
           )


def split_plot(data, methods, lower_bound=0.85):
    x = np.arange(1,6)

    cmap = colourmap(methods) #tolist bc theyy are packed as np arrays
    mmap = markermap(methods)
    lmap = linestylemap(methods)

    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['lines.marker'] = "s"

    plt.clf()
    g, ax1 = plt.subplots(1, 6, figsize=(16,2), sharex=True,sharey=True)
    h, ax2 = plt.subplots(1, 6, figsize=(16,2), sharex=True,sharey=True)

    for acc in data:
        for ax in [ax1, ax2]:
            for i in range(5):
                ax[i].plot(x[i:], acc[i:, i])
            ax[-1].plot(x, np.nanmean(acc,0))

    for i in range(6):
        for ax in [ax1, ax2]:
            for j, line in enumerate(ax[i].get_lines()):
                line.set_color(cmap[j])
                line.set_linestyle(lmap[j])
                line.set_marker(mmap[j])
        
    for i in range(6):
        # ax1[i].set_yticks(np.arange(0.25, 1.25, 0.25))
        # ax2[i].set_yticks(np.arange(0.85, 1.01, 0.05))
        # if lower_bound != 0.85:
        #     ax2[i].set_yticks(np.arange(lower_bound, 1.01, (1-lower_bound)/4))
        for ax in [ax1, ax2]:
            ax[i].set_xticks(np.arange(1, 6, 1))
            ax[i].title.set_text(f"Task {i+1}")
            ax[i].set_xlabel("Task"); 
            ax[i].set_ylabel("Accuracy"); 
        plt.ylim(lower_bound,1.01)

            

    ax1[5].title.set_text("Average"); ax2[5].title.set_text("Average")

    g.legend(   # The line objects
           labels=methods,
           loc="center right",   # Position of legend
           borderaxespad=0.3,    # Small spacing around legend box
           title="Methods:"  # Title for the legend
           )
    h.legend(   # The line objects
           labels=methods,
           loc="center right",   # Position of legend
           borderaxespad=0.3,    # Small spacing around legend box
           title="Methods:"  # Title for the legend
           )

def three_row_plot_coresets(data, methods, lower_bound=None):
    x = np.arange(1,6)

    cmap = colourmap(methods) #tolist bc theyy are packed as np arrays
    mmap = markermap(methods)
    lmap = linestylemap(methods)

    mpl.rcParams['lines.markersize'] = 4
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['lines.marker'] = "s"

    plt.clf()
    g, ax1 = plt.subplots(1, 6, figsize=(16,2), sharex=True,sharey=True)
    h, ax2 = plt.subplots(1, 6, figsize=(16,2), sharex=True,sharey=True)
    j, ax3 = plt.subplots(1, 6, figsize=(16,2), sharex=True,sharey=True)

    data = [data, data[:-3], data[-3:]]
    methods = [methods, methods[:-3], methods[-3:]]

    for j, ax in enumerate([ax1, ax2, ax3]):
        for acc in data[j]:
            for i in range(5):
                ax[i].plot(x[i:], acc[i:, i])
            ax[-1].plot(x, np.nanmean(acc,0))

    for i in range(6):
        for k, ax in enumerate([ax1, ax2, ax3]):
            cmap = colourmap(methods[k]) #tolist bc theyy are packed as np arrays
            mmap = markermap(methods[k])
            lmap = linestylemap(methods[k])
            for j, line in enumerate(ax[i].get_lines()):
                line.set_color(cmap[j])
                line.set_linestyle(lmap[j])
                line.set_marker(mmap[j])
        
    for i in range(6):
        # ax1[i].set_yticks(np.arange(0.25, 1.25, 0.25))
        # ax2[i].set_yticks(np.arange(0.85, 1.01, 0.05))
        # if lower_bound != 0.85:
        #     ax2[i].set_yticks(np.arange(lower_bound, 1.01, (1-lower_bound)/4))
        for j, ax in enumerate([ax1, ax2, ax3]):
            ax[i].set_xticks(np.arange(1, 6, 1))
            ax[i].title.set_text(f"Task {i+1}")
            ax[i].set_xlabel("Task"); 
            ax[i].set_ylabel("Accuracy"); 
            if lower_bound is not None:
                ax[i].set_ylim(lower_bound[j], 1.01)
        

    ax1[5].title.set_text("Average"); ax2[5].title.set_text("Average"); ax3[5].title.set_text("Average")

    g.legend(   # The line objects
           labels=methods[0],
           loc="center right",   # Position of legend
           borderaxespad=0.3,    # Small spacing around legend box
           title="Methods:"  # Title for the legend
           )
    # h.legend(   # The line objects
    #        labels=methods[1],
    #        loc="center right",   # Position of legend
    #        borderaxespad=0.3,    # Small spacing around legend box
    #        title="Methods:"  # Title for the legend
    #        )

# Formatting funcitonality
def colourmap(alist):
    col = []
    for i, a in enumerate(alist):
        a = a.lower()
        if "qr" in a:
            col.append("tab:red")
        elif "vcl" == a:
            col.append("tab:blue")
        elif "k-center" in a:
            col.append("tab:green")
        elif "random" in a:
            col.append("tab:purple")
        
    return col

def linestylemap(alist):
    col = []
    for i, a in enumerate(alist):
        if "coreset" in a.lower():
            col.append("dashed")
        else:
            col.append("solid")
    return col

def markermap(alist):
    col = []
    for i, a in enumerate(alist):
        if "coreset" in a.lower():
            col.append("s")
        else:
            col.append("o")
    return col