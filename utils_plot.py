from re import S



def compareShapleyValues(
    sh_score_1,
    sh_score_2,
    toOrder=0,
    title=[],
    shared_x=False,
    height=0.8,
    linewidth=0.8,
    sizeFig=(7, 7),
    saveFig=False,
    nameFig=None,
    labelsize=10,
    subcaption=False,
    pad=None,
    show_figure=True,
):
    import matplotlib.pyplot as plt

    sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
    sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}
    if toOrder == 0:
        sh_score_1 = {
            k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1])
        }
        sh_score_2 = {k: sh_score_2[k] for k in sorted(sh_score_1, key=sh_score_1.get)}
    elif toOrder == 1:
        sh_score_2 = {
            k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1])
        }
        sh_score_1 = {k: sh_score_1[k] for k in sorted(sh_score_2, key=sh_score_2.get)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sizeFig, dpi=100, sharex=shared_x)


    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )

    ax1.set_yticks(range(len(sh_score_1)), minor=False)
    ax1.set_yticklabels(list(sh_score_1.keys()), fontdict=None, minor=False)

    if len(title) > 1:
        ax1.set_title(title[0])
    
    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="#7CBACB",
        height=height,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    # plt.yticks(range(len(sh_score_2)), [])
    ax2.set_yticks(range(len(sh_score_1)), minor=False)
    ax2.set_yticklabels([], minor=False)
    if len(title) > 1:
        ax2.set_title(title[1])


    ax1.tick_params(axis="y", labelsize=labelsize)
    if pad:
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout()

    s1 = "(a)" if subcaption else ""  # r"$\bf{(a)}$"
    s2 = "(b)" if subcaption else ""  # r"$\bf{(b)}$"
    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    if saveFig:
        nameFig = "./shap.pdf" if nameFig is None else f"{nameFig}.pdf"
        plt.savefig(nameFig, format="pdf", bbox_inches="tight")

    if show_figure:
        plt.show()
        plt.close()


    
def plot_info_by_name(results, name_info = "MSE", figsize=(8,5), show_figure=True,  title_additional = ""):

    """
    Plot the results of the mitigation

    Args:
        results (pd.DataFrame) : results of mitigation
        name_info (str): information to plot
        figsize (tuple)
        show_figure (bool): if True, show the figure
        title_additional (str): add title_additional to the title
    """
    
    info_dict = dict(results[name_info])

    completely_mitigated = results.loc[results["#negdiv"]==0]
    if completely_mitigated.shape[0]>0:
        first_not_divergent = completely_mitigated.iloc[0].name
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(info_dict.keys(), info_dict.values(), marker = ".")
    if completely_mitigated.shape[0]>0:
        ax.scatter(first_not_divergent, info_dict[first_not_divergent], c = "r")
    ax.set_xlabel("#Iterations")
    title = title_additional + name_info
    ylabel = name_info
    if name_info == "MSE":
        ylabel = "MSE"
        title = title_additional + r"Mean Squared Error"
    elif name_info == "min_alpha":
        ylabel = "α - Minimum divergence"
        title = title_additional + r"Mininum divergence α -  min(Δ($g_i$))"
        
    ax.set_ylabel(ylabel)
    plt.title(title)
    if show_figure:
        plt.show()

