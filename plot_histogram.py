from relative_error import relative_error
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_comp_methods(title, is_normed, interp, xlim = [1e-3, 1e+1], ylim = [0, 1e-1], x_range_log = [-4, 3], is_cum = False, loc = "best"):
    bins = np.logspace(x_range_log[0], x_range_log[1], (x_range_log[1] - x_range_log[0]) * 20, base = 10)
    fontsize = 26
    labelsize = 15
    length_major, length_minor = 20, 10
    direction = "in"
    interp_methods = list(interp.keys())
    m_list = list(interp[interp_methods[0]].keys())
    cmap = plt.get_cmap("tab10")
    color = {}
    for i, ip_key in enumerate(interp_methods):
        color[ip_key] = cmap(i)
    
    for m_key in m_list:
        fig = plt.figure(figsize = (8, 5))
        ax_hist = fig.add_subplot(111)
        ax_hist.set_xscale("log")
        ax_hist.set_xlabel("Relative Error", fontsize = fontsize)
        ax_hist.set_ylabel("Relative Frequency", fontsize = fontsize)
        ax_hist.set_xlim(xlim)
        ax_hist.set_ylim(ylim)
        ax_hist.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
        ax_hist.tick_params(labelsize = labelsize, length = length_minor, direction = direction, which = "minor")
        if is_cum:
            ax_cum = ax_hist.twinx()
            ax_cum.set_ylabel("Cumulative", fontsize = fontsize)
            ax_cum.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
        
        for ip_key in interp_methods:
            original, predict = [], []
            if ip_key == "origin":
                continue
            for i in range(len(interp[ip_key][m_key])):
                mask = (interp["origin"][m_key][i] != interp[ip_key][m_key][i])
                original += list(interp["origin"][m_key][i][mask].reshape(-1))
                predict += list(interp[ip_key][m_key][i][mask].reshape(-1))
            original = np.array(original)
            predict = np.array(predict)
            error = relative_error(original, predict)
            weights = np.ones_like(error) / error.size
            hist_num, hist_bins, _ = ax_hist.hist(error, bins = bins, weights = weights, edgecolor = color[ip_key], histtype = "step", linewidth = 2.5, label = "Relative Freq({})".format(ip_key))
            if is_cum:
                ax_cum.plot(hist_bins[1:], hist_num.cumsum(), color = color[ip_key], label = "Cumulative({})".format(ip_key))
        
        if is_cum:
            handler1, label1 = ax_hist.get_legend_handles_labels()
            handler2, label2 = ax_cum.get_legend_handles_labels()
            ax_hist.legend(handler1 + handler2, label1 + label2, loc = loc, borderaxespad = 0., fontsize = int(fontsize * 0.6))
        else:
            ax_hist.legend(loc = loc, fontsize = int(fontsize * 0.6))
        title_ = title
        if is_normed:
            title_ += "({}_normed)".format(m_key[11:16])
        else:
            title_ += "({})".format(m_key[11:16])
        plt.title(title_, fontsize = fontsize)
        plt.show()


def plot_histogram_machine_learning_models(is_normed, interp, xlim = [1e-3, 1e+1], ylim = [0, 1e-1], x_range_log = [-4, 3], is_cum = False, loc = "best"):
    bins = np.logspace(x_range_log[0], x_range_log[1], (x_range_log[1] - x_range_log[0]) * 20, base = 10)
    fontsize = 26
    labelsize = 15
    length_major, length_minor = 20, 10
    direction = "in"
    caption_list = list(interp.keys())
    m_list = list(interp[caption_list[0]]["origin"].keys())
    cmap = plt.get_cmap("tab10")
    color = {}
    for i, caption in enumerate(caption_list):
        color[caption] = cmap(i)
        
    for m_key in m_list:
        fig = plt.figure(figsize = (8, 5))
        ax_hist = fig.add_subplot(111)
        ax_hist.set_xscale("log")
        ax_hist.set_xlabel("Relative Error", fontsize = fontsize)
        ax_hist.set_ylabel("Relative Frequency", fontsize = fontsize)
        ax_hist.set_xlim(xlim)
        ax_hist.set_ylim(ylim)
        ax_hist.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
        ax_hist.tick_params(labelsize = labelsize, length = length_minor, direction = direction, which = "minor")
        if is_cum:
            ax_cum = ax_hist.twinx()
            ax_cum.set_ylabel("Cumulative", fontsize = fontsize)
            ax_cum.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
            
        for caption in caption_list:
            original, predict = [], []
            for i in range(len(interp[caption]["origin"][m_key])):
                mask = (interp[caption]["origin"][m_key][i] != interp[caption]["ann"][m_key][i])
                original += list(interp[caption]["origin"][m_key][i][mask].reshape(-1))
                predict += list(interp[caption]["ann"][m_key][i][mask].reshape(-1))
            original = np.array(original)
            predict = np.array(predict)
            error = relative_error(original, predict)
            weights = np.ones_like(error) / error.size
            hist_num, hist_bins, _ = ax_hist.hist(error, bins = bins, weights = weights, edgecolor = color[caption], histtype = "step", linewidth = 2.5, label = "Relative Freq({})".format(caption))
            if is_cum:
                ax_cum.plot(hist_bins[1:], hist_num.cumsum(), color = color[caption], label = "Cumulative({})".format(caption))
        if is_cum:
            handler1, label1 = ax_hist.get_legend_handles_labels()
            handler2, label2 = ax_cum.get_legend_handles_labels()
            ax_hist.legend(handler1 + handler2, label1 + label2, loc = loc, borderaxespad = 0., fontsize = int(fontsize * 0.6))
        else:
            ax_hist.legend(loc = loc, fontsize = int(fontsize * 0.6))
        if is_normed:
            title_ = "({}_normed)".format(m_key[11:16])
        else:
            title_ = "({})".format(m_key[11:16])
        plt.title(title_, fontsize = fontsize)
        plt.show()
