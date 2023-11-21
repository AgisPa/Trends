import matplotlib.pyplot as plt
import numpy as np
import math

plt.style.use('Solarize_Light2')
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.set_facecolor((.18, .31, .31))



def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def plot_custom_graphs(preds, preds_up, preds_down, im_rs, uppers, lowers, opens, volat, max_accuracy, accuracy,interest):

    max_accuracy=interest

    axs[0, 1].plot(np.linspace(0,len(preds),len(preds)),preds, color=rgb_to_hex(255, (204 - int(204 * max_accuracy)), 255), alpha=0.2*interest)
    axs[0,1].set_title("LS Prediction")
    axs[0, 0].set_title("Normal Distribution Around Spot")
    axs[1, 0].set_title("Critical Points")
    axs[1, 1].set_title("PLs Prediction")
    axs[0, 1].fill_between(np.linspace(int(len(preds) / accuracy), len(preds),
                                       len(preds)), preds, preds_up,
                           color=rgb_to_hex(255, 255, (204 - int(204 * max_accuracy))), alpha=0.1 * interest)

    axs[0, 1].fill_between(np.linspace(int(len(preds) / accuracy), len(preds),
                                       len(preds)), preds_down, preds,
                           color=rgb_to_hex((204 - int(204 * max_accuracy)), 255, 255),
                           alpha=0.1 * interest)
    for j  in range(len(im_rs)):
        if im_rs[j]==0:
            im_rs[j]=im_rs[j-1]
    for j  in range(len(lowers)):
        if lowers[j]==0:
            lowers[j]=lowers[j-1]
    for j  in range(len(uppers)):
        if uppers[j]==0:
            uppers[j]=uppers[j-1]
    axs[0, 0].fill_between(np.linspace(0, len(im_rs),len(im_rs)), im_rs, uppers, color="green",
                           alpha=0.4 * max_accuracy)
    axs[0, 0].fill_between(np.linspace(0, len(im_rs), len(im_rs)), lowers, im_rs, color="red", alpha=0.4 * interest)

    ins_vol = axs[1,0].inset_axes([0, 0, 1, 0.25], sharex=axs[1,0])
    ins_vol.fill_between(np.linspace(0, len(volat), len(volat)), volat,
                         color=rgb_to_hex((18 ), 30, (225 - int(125 * max_accuracy))),
                          alpha=0.1*interest)
    ins_vol.patch.set_alpha(0)
    ins_vol.axis("off")

    axs[0, 0].plot(im_rs, linestyle="dashed", color=rgb_to_hex(144, 144, 144))
    axs[0, 0].plot(opens[:len(im_rs)], color="black")
    axs[0, 0].plot(opens[len(im_rs)-1:], color="gray")

    axs[0, 1].plot(opens[:len(im_rs)], color="black")
    axs[0, 1].plot(opens[len(im_rs)-1:], color="gray")

    axs[1,0].plot(opens[:len(im_rs)], color="black")
    axs[1,0].plot(opens[len(im_rs) - 1:], color="gray")
    axs[1,1].plot(opens[:len(im_rs)], color="black")
    axs[1, 1].plot(opens[len(im_rs)-1:], color="gray")

    if max_accuracy == 1:
        axs[0, 1].plot(np.linspace(0, len(preds),
                                   len(preds)), preds, label="Predicted Mean",
                       color=rgb_to_hex(255, (204 - int(204 * max_accuracy)), 255),
                       alpha=0.2 * interest)
        axs[0, 1].fill_between(np.linspace(int(len(preds) / accuracy), len(preds),
                                           len(preds)), preds,
            preds_up, color=rgb_to_hex(255, 255, (204 - int(204 * max_accuracy))), alpha=interest,
            label="Expected Upper Boundary")
        axs[0, 1].fill_between(
            np.linspace(int(len(preds) / accuracy), len(preds),
                        len(preds)),
            preds_down, preds, color=rgb_to_hex((204 - int(204 * max_accuracy)), 255, 255), alpha=interest,
            label="Expected Lower Boundary")

        axs[0, 0].fill_between(np.linspace(0, len(im_rs), len(im_rs)), im_rs, uppers, color="green", alpha=interest,
                               label="N(r,sigma) Upper Boundary")
        axs[0, 0].fill_between(np.linspace(0, len(im_rs), len(im_rs)), lowers, im_rs, color="red", alpha=interest,
                               label="N(r,sigma) Lower Boundary")

        ins_vol=axs[1,0].inset_axes([0, 0, 1, 0.25], sharex=axs[1,0])
        ins_vol.fill_between(np.linspace(0, len(im_rs), len(im_rs)), volat,
                               color=rgb_to_hex((125 - int(125 * max_accuracy)), 30, (255 - int(105 * max_accuracy))),
                               label="Variance", alpha=interest)
        ins_vol.patch.set_alpha(0)

        ins_vol.axis("off")

        
        axs[1, 0].set_title("Cross Points and Volatility")

        axs[0, 0].set_title("N(rms,sigma)")
        axs[0, 1].set_title("Predicted")
        axs[1,1].set_title("Variance")

        point_up_lower = \
            crossing_points(preds_up,
                            lowers)[0]
        value_up_lower = \
            crossing_points(preds_up,
                            lowers)[1]

        point_down_upper = \
            crossing_points(preds_down,
                            uppers)[
                0]
        value_down_upper = \
            crossing_points(preds_down,
                            uppers)[
                1]
        ins = axs[1, 0].inset_axes([0, 0, 1, 0.25], sharex=axs[1, 0])
        ins.scatter(point_up_lower, np.multiply(value_up_lower,-1), marker="x", color="red", s=3, label="Upper Prediction Lower Boundary")
        ins.scatter(point_down_upper, value_down_upper, marker="x", color="green", s=3, label="Lower Prediction Upper Boundary")
        ins.patch.set_alpha(0)
        ins.axis("off")
        ins.legend(bbox_to_anchor=(1, 4),fontsize=7)
        ins_vol.legend(bbox_to_anchor=(1, 4),fontsize=7)

        axs[1,0].legend(fontsize=7, loc="upper left")
        axs[0,1].legend(fontsize=7, loc="upper left")
        axs[1,1].legend(fontsize=7, loc="upper left")
        axs[0, 0].legend(fontsize=7, loc="upper left")





    point_up_lower = \
        crossing_points(preds_up,
                        lowers)[0]
    value_up_lower = \
        crossing_points(preds_up,
                        lowers)[1]

    point_down_upper = \
        crossing_points(preds_down,
                        uppers)[
            0]
    value_down_upper = \
        crossing_points(preds_down,
                        uppers)[
            1]
    ins=axs[1,0].inset_axes([0, 0, 1, 0.25], sharex=axs[1,0])
    ins.scatter(point_up_lower,np.multiply(value_up_lower,-1), marker="x", color="red", s=3)
    ins.scatter(point_down_upper, value_down_upper, marker="x", color="green", s=3)
    ins.patch.set_alpha(0)
    ins.axis("off")
    return fig, axs




def crossing_points(x, y):
    points = []
    values = []

    while len(x) > len(y):
        y.extend([math.nan])
    while len(y) > len(x):
        x.extend([math.nan])
    check = -1
    for i in range(0, len(x)):
        if x[i] != math.nan and y[i] != math.nan:
            if np.sign(x[i] - y[i]) != np.sign(check):
                points.append(i)
                values.append(abs(x[i]-y[i]))
            check = x[i] - y[i]


    return [points, values]




