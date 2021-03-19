from relative_error import relative_error
import matplotlib.pyplot as plt
import numpy as np

def plot_data_error_vs_scale(origin, ann, linear, cubic, scale_factor, param_kind, superior_ratio, error_ylim = [1e-3, 1e+1]):
    eps = 1e-7
    error_ann = relative_error(origin, ann)
    error_linear = relative_error(origin, linear)
    error_cubic = relative_error(origin, cubic)
    eratio_linear_to_ann = np.abs(error_linear / (error_ann + eps))
    eratio_cubic_to_ann = np.abs(error_cubic / (error_ann + eps))
    cmask_linear = (eratio_linear_to_ann >= superior_ratio)
    cmask_cubic = (eratio_cubic_to_ann >= superior_ratio)
    width = scale_factor[1] - scale_factor[0]
    color_linear, color_cubic = "green", "red"
    align = "center"
    loc = "upper right"
    linewidth = 2.5
    fontsize = 26
    labelsize = 13
    length_major = 16
    length_minor = 8
    direction = "in"
    
    fig = plt.figure(figsize = (15, 6))
    ##ANN data.
    ax_ann_d = fig.add_subplot(231)
    ax_ann_d.plot(scale_factor, ann, label = "ANN", linewidth = linewidth)
    ax_ann_d.plot(scale_factor, origin, label = "Origin", linewidth = linewidth)
    ##ANN error.
    ax_ann_e = fig.add_subplot(234)
    ax_ann_e.bar(scale_factor, error_ann, width = width, align = align, label = "ANN")
    if not np.all(cmask_linear == False):
        ax_ann_e.bar(scale_factor[cmask_linear], error_ann[cmask_linear], label = "ANN > {}*Linear".format(superior_ratio), width = width, align = align, color = color_linear)
    if not np.all(cmask_cubic == False):
        ax_ann_e.bar(scale_factor[cmask_cubic], error_ann[cmask_cubic], label = "ANN > {}*Cubic".format(superior_ratio), width = width, align = align, color = color_cubic)
    ax_ann_e.set_yscale("log")
    ax_ann_e.set_ylim(error_ylim)
    ax_ann_d.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
    ax_ann_e.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
    ax_ann_e.tick_params(labelsize = labelsize, length = length_minor, direction = direction, which = "minor")
    ##Linear data.
    ax_linear_d = fig.add_subplot(232)
    ax_linear_d.plot(scale_factor, linear, label = "Linear", linewidth = linewidth)
    ax_linear_d.plot(scale_factor, origin, label = "Origin", linewidth = linewidth)
    ##Linear error.
    ax_linear_e = fig.add_subplot(235)
    ax_linear_e.bar(scale_factor, error_linear, width = width, align = align, label = "Linear")
    if not np.all(cmask_linear == False):
        ax_linear_e.bar(scale_factor[cmask_linear], error_linear[cmask_linear], label = "ANN > {}*linear".format(superior_ratio), width = width, align = align, color = color_linear)
    ax_linear_e.set_yscale("log")
    ax_linear_e.set_ylim(error_ylim)
    ax_linear_d.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
    ax_linear_e.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
    ax_linear_e.tick_params(labelsize = labelsize, length = length_minor, direction = direction, which = "minor")
    ##Cubic data.
    ax_cubic_d = fig.add_subplot(233)
    ax_cubic_d.plot(scale_factor, cubic, label = "Cubic", linewidth = linewidth)
    ax_cubic_d.plot(scale_factor, origin, label = "Origin", linewidth = linewidth)
    ##Cubic error.
    ax_cubic_e = fig.add_subplot(236)
    ax_cubic_e.bar(scale_factor, error_cubic, width = width, align = align, label = "Cubic")
    if not np.all(cmask_cubic == False):
        ax_cubic_e.bar(scale_factor[cmask_cubic], error_linear[cmask_cubic], label = "ANN > {}*Cubic".format(superior_ratio), width = width, align = align, color = color_cubic)
    ax_cubic_e.set_yscale("log")
    ax_cubic_e.set_ylim(error_ylim)
    ax_cubic_d.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
    ax_cubic_e.tick_params(labelsize = labelsize, length = length_major, direction = direction, which = "major")
    ax_cubic_e.tick_params(labelsize = labelsize, length = length_minor, direction = direction, which = "minor")
    
    ax_ann_d.set_ylabel(param_kind, fontsize = fontsize)
    ax_ann_e.set_ylabel("Relative Error", fontsize = fontsize)
    ax_ann_e.set_xlabel("Scale Factor", fontsize = fontsize)
    ax_linear_e.set_xlabel("Scale Factor", fontsize = fontsize)
    ax_cubic_e.set_xlabel("Scale Factor", fontsize = fontsize)
    ax_ann_d.legend(loc = loc, fontsize = int(fontsize * 0.6))
    ax_linear_d.legend(loc = loc, fontsize = int(fontsize * 0.6))
    ax_cubic_d.legend(loc = loc, fontsize = int(fontsize * 0.6))
    ax_ann_e.legend(loc = loc, fontsize = int(fontsize * 0.6))
    ax_linear_e.legend(loc = loc, fontsize = int(fontsize * 0.6))
    ax_cubic_e.legend(loc = loc, fontsize = int(fontsize * 0.6))
    plt.show()
