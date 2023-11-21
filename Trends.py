import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv
import warnings
import Plotting
from scipy.optimize import minimize
import tkinter as tk
from tkinter import ttk
import time
from alive_progress import alive_bar
from alive_progress import config_handler
import timeit
import pigar
pigar



config_handler.set_global(length=60, spinner="radioactive", bar="bubbles", force_tty=True)


# warnings.simplefilter("ignore")
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line

def propability(data):
    if len(data) == 1:
        r = data[0]
        sigma = 1
    else:
        r = sum([data[i] for i in range(0, len(data))]) / len(data)
        sigma = np.sqrt(sum([(data[i] - r) ** 2 for i in range(0, len(data))]) / (len(data) - 1))
    if sigma < 1:
        sigma = 1
    pxs = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((abs(data[i] - r) / sigma) ** 2) / 2) * data[i] ** 2 for i in
           range(0, len(data))]
    norm = sum(pxs)

    pxs = pxs / norm

    return pxs, sigma, r


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def least_sq(rs):
    rs = [i for i in rs if i != 0]
    A = np.vstack([np.linspace(0, len(rs), len(rs)), np.ones(len(rs))]).T

    rs_t = np.array(rs)[:, np.newaxis]
    alpha, beta = np.linalg.lstsq(A, rs_t, rcond=1)[0]
    return alpha[0], beta[0]


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def partitioned_ls_alternating(y, P, T, interest):
    X = np.linspace(0, len(y), len(y))
    parts = np.array(P).shape[0]

    alpha = np.random.randint(0, len(y), len(y))
    bg = y[0]
    yx = np.array([i - bg for i in y])

    def b_con(beta):
        bbs = list(chunks(beta, parts))
        step = int(len(beta) / parts)

        for i in range(0, step):
            bbs[i] = sum(list(bbs[i])) / parts

        bb = []
        beta = list(beta)
        for i in range(0, step):
            for j in range(i * parts, (i + 1) * parts):
                bb.append(bbs[i])

        add = [bb[o] - beta[o] for o in range(0, len(bb))]
        results = np.sum(add)

        return results / 100000000

    res_a = []

    res_b = []
    alpha = np.random.rand(len(y))
    alpha = [alpha[p] / max(alpha) * interest for p in range(0, len(y))]
    P = np.array(P)
    beta = np.random.randint(-1, 1, len(y))
    for _ in range(T):
        if _ == 0:
            res_a.append(alpha)

        def min_squares_alpha(alpha):
            return abs(np.sum(
                [np.subtract(np.multiply(np.multiply(np.multiply(X, P[i]), alpha), beta), yx) ** 2 for i in
                 range(0, parts)])) / 1000000000

        def min_squares_beta(beta):
            return abs(np.sum(
                [np.subtract(np.multiply(np.multiply(np.multiply(X, P[i]), alpha), beta), yx) ** 2 for i in
                 range(0, parts)])) / 1000000000

        if _ == 0:
            res_b.append(beta)
        cons_b = ({'type': 'eq', 'fun': lambda beta: b_con(beta)}
        )
        result_b = minimize(lambda beta: min_squares_beta(beta), res_b[len(res_b) - 1], method='SLSQP',
                            constraints=cons_b)
        res_b.append(result_b.x)

        cons_a = ({'type': 'eq', 'fun': lambda alpha: sum(alpha) - 1}
        )
        result_a = minimize(lambda alpha: min_squares_alpha(alpha), res_a[len(res_a) - 1], method='BFGS')
        res_a.append(result_a.x)

        alpha = res_a[len(res_a) - 1]
        beta = res_b[len(res_b) - 1]

    return alpha, beta


def list_prop_length(a, x0, steps, interest, size):
    for i in range(0, len(x0) + (interest) * steps):
        if i == len(a):
            a.append(get_data(a, interest, steps, interest, size)["preds"][len(a) - 1])
    return a


def partitioned_least_squares(a, part, T, steps, interest, size):
    a = list(a)
    x0 = a.copy()
    a = list_prop_length(a, x0, steps, interest, size)

    result = [[0 for i in range(len(a))] for j in range(0, part)]
    for i in range(0, part):
        for j in range(0, len(a) - 1):
            if int(i * (len(a) / part)) == j:
                for l in range(0, int(len(a) / part)):
                    result[i][j + l] = 1
    asd = []
    b = []
    if T != 0:
        approx_alter = partitioned_ls_alternating(a, result, T, interest)
        asd = approx_alter[0]
        b = approx_alter[1]

    return asd, b, [a[l] for l in range(0, len(x0) + interest * steps)], interest, x0


def get_data(opens, increment, future_steps, interest, size):
    if len(opens) // increment == len(opens) / increment:
        accuracy = int(increment)
        limit = 0
    else:
        accuracy = int(increment)
        limit = 1
    step = int(size / accuracy)
    prs = np.zeros(len(opens))
    rs = np.zeros(size)
    sigmas = np.zeros(size)
    pred = np.zeros(size + int(future_steps * interest))
    pred_up = np.zeros(size + int(future_steps * interest))
    pred_down = np.zeros(size + int(future_steps * interest))
    upper = np.zeros(size)
    lower = np.zeros(size)

    for k in range(1, size + future_steps * increment + 1):

        if k / accuracy == int(
                k / accuracy) and k <= size + future_steps * increment or k == size + future_steps * interest:
            start = k - accuracy
            for l in range(0, accuracy):
                if k < size:
                    stats = propability(opens[start:start + l + 1])
                    r = stats[2]
                    sigma = stats[1]
                    sigmas[k - accuracy + l] = sigma
                    rs[k - accuracy + l] = r
                    upper[k - accuracy + l] = r + sigma
                    lower[k - accuracy + l] = r - sigma
                    pred[k - accuracy + l] = least_sq(rs)[0] * k + least_sq(rs)[1]
                    pred_up[k - accuracy + l] = least_sq(upper)[0] * k + least_sq(upper)[1]
                    pred_down[k - accuracy + l] = least_sq(lower)[0] * k + least_sq(lower)[1]

                elif size <= k <= size + future_steps * interest:
                    pred[k - accuracy + l] = least_sq(rs)[0] * k + least_sq(rs)[1]
                    pred_up[k - accuracy + l] = least_sq(upper)[0] * k + least_sq(upper)[1]
                    pred_down[k - accuracy + l] = least_sq(lower)[0] * k + least_sq(lower)[1]

    preds_up = list(pred_up)
    preds_down = list(pred_down)
    im_rs = list(rs)
    preds = list(pred)
    volat = list(sigmas)
    uppers = list(upper)
    lowers = list(lower)

    return {"preds_up": preds_up, "preds_down": preds_down, "preds": preds, "volat": volat, "uppers": uppers,
            "lowers": lowers,
            "im_rs": im_rs, "opens": opens, "interest": interest}


def bin(x):
    if x:
        return 1
    if not x:
        return 0


def probability_density(interest, steps, size, training, ml_steps, path, Plots, start):
    if interest == 1:
        warnings.warn("Partition size must be bigger that one.")
    # Change below for upper limit of partition number as a multiple of the interest.
    max_levels = int(2 * interest)

    # Change below for the minimum partition number as a multiple of the interest.
    increments = int(0.5 * interest)


    if steps*interest<size-training:
        steps=int((size-training)/interest)

    if path == "BTC-USD.csv":
        data = pd.read_csv(path, nrows=size)
        data.reindex(index=data.index[::-1])
        opens = np.array(data["Open"][:training])
        data=data["Open"]
    else:
        data = pd.read_csv(path, nrows=size)

        if path=="example.csv":
            data.reindex(index=data.index[::-1])
            data = data[str(data.keys().values.tolist()[1])].tolist()
            opens=np.array(data[:training])
            # Add column name for the trend data with the following command.
            # opens=np.array(data["NAME_OF_COLUMN"][:training])
        else:
            data.reindex(index=data.index[::-1])
            data = data[str(data.keys().values.tolist()[1])].tolist()
            opens = np.array(data[:training])
            if len(data[:training]) != training:
                print("Add column name for trend data analysis at line 242.")



    ls_output=[]
    pls_output=[]
    weight=[]

    f_list = []                        
    x0 = opens.copy()
    incs = []

    for i in range(2, training):
        if i >= increments and training / i > 2 and int(training / i) == training / i or i == interest:
            incs.append(i)
    print("Contributing partitions:", set(incs), "for time frame of interest equal to", interest)

    if max_levels == "max":
        max_levels = len(opens)

    future_steps = interest * steps

    last_pos_ls = []
    last_pos_Pls = []
    approx_label = []

    prtn = 0
    total = len(incs)
    with alive_bar(total) as bar:
        for i in range(increments, training + steps * future_steps + 1):

            if training / i > 2 and int(training / i) == training / i or i == interest:

                time.sleep(0.1)

                f_list.append((np.sqrt(2) / np.sqrt(np.pi) * np.exp(-(abs(i - int((interest))) / interest) ** 2 / 2)))
                f_round = (f_list[len(f_list) - 1])

                approx = list(partitioned_least_squares(x0, i, ml_steps, steps, interest, training))
                part_prox_a = list(approx)[0]
                part_prox_b = list(approx)[1]
                part_prox = []
                if i < size:

                    final_pos = approx[2]
                else:
                    final_pos = x0
                for j in range(0, len(final_pos)):
                    part_prox.append(part_prox_a[j] * j * part_prox_b[j] + final_pos[j])

                opens_data = get_data(final_pos[:training + steps * interest], i, future_steps, interest, training)
                x0_data = get_data(x0[:training], i, future_steps, interest, len(x0))
                ls_preds = opens_data["preds"][:len(x0) + steps * interest]
                ls_preds_down = opens_data["preds_down"][:len(x0) + steps * interest]
                ls_preds_up = opens_data["preds_up"][:len(x0) + steps * interest]
                part_average = x0_data["im_rs"]
                upper_limit = x0_data["uppers"]
                lower_limit = x0_data["lowers"]
                volat = x0_data["volat"]
                ls_output.append(ls_preds)
                pls_output.append(part_prox)

                if Plots == "Pls" or Plots == "full" or Plots=="None":
                    if approx_label.count("Algorith based on partitions of " + str(i)) == 0:
                        approx_label.append("Algorith based on partitions of " + str(i))
                        plt.plot(part_prox, alpha=f_round, label=approx_label[len(approx_label) - 1])
                    if prtn == 0:
                        prtn += 1





                    else:
                        plt.plot(opens[:len(x0)], color="black")
                        plt.plot(opens[len(x0):], color="gray")
                        plt.plot(part_prox, alpha=f_round)
                if Plots == "full" or Plots == "Ls" or Plots=="None":


                    axs = Plotting.plot_custom_graphs(preds=ls_preds,
                                                      preds_up=ls_preds_up,
                                                      preds_down=ls_preds_down,
                                                      im_rs=part_average,
                                                      uppers=upper_limit,
                                                      lowers=lower_limit,
                                                      opens=data,
                                                      volat=volat, max_accuracy=max_levels, accuracy=i,
                                                      interest=f_round)
                last_pos_Pls.append(part_prox[len(part_prox) - 1])
                last_pos_ls.append(opens_data["preds"][training + steps * interest - 1])

                bar()

    f_list = [f_list[j] / sum(f_list) for j in range(0, len(f_list))]
    result_ls = sum([f_list[k] * last_pos_ls[k] for k in range(0, len(f_list))])
    result_Pls = sum([f_list[k] * last_pos_Pls[k] for k in range(0, len(f_list))])
    if Plots == "full":
        plt.scatter(x=len(opens) + steps * interest, y=result_Pls, label='PLs Learning Prediction')
        plt.scatter(x=len(opens) + steps * interest, y=result_ls, label='Least Squares Prediction')
        print("PLs Learning Predictions in the ", steps * interest, " day time frame :", result_Pls,
              ", the least squares predictions ", result_ls)
    if Plots == "Pls":
        plt.scatter(x=len(opens) + steps * interest, y=result_Pls, label='PLs Learning Prediction')
        print("The partitioned least squares predictions for the", steps * interest, " day time frame  ", result_Pls)
    if Plots == "Ls":
        plt.scatter(x=len(opens) + steps * interest, y=result_ls, label='Least Squares Prediction')
        print("The least squares predictions for the", steps * interest, " day time frame  ", result_ls)

    plt.suptitle("Analysis")

    plt.legend()

    output ={"ls_predictions": ls_output, "pls_predictions": pls_output,"weight":f_list}
    output=pd.DataFrame([output])
    output.to_csv("Trends.csv")
    stop = timeit.default_timer()


    print('Time: ', stop - start)

    if Plots!="None":
        plt.show()


class pop_up_exe(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.geometry("500x220")

        e1 = tk.Entry(self)
        e2 = tk.Entry(self)
        e3 = tk.Entry(self)
        e4 = tk.Entry(self)
        e5 = tk.Entry(self)
        e6 = tk.Entry(self)
        e7 = tk.Entry(self)

        e1.grid(row=0, column=1)
        e2.grid(row=1, column=1)
        e3.grid(row=2, column=1)
        e4.grid(row=3, column=1)
        e5.grid(row=4, column=1)
        e6.grid(row=5, column=1)
        e7.grid(row=6, column=1)

        e1.insert(0, "sample")
        e2.insert(1, 50)
        e3.insert(2, 5)
        e4.insert(3, 0)
        e5.insert(4, 80)
        e6.insert(5, 20)
        e7.insert(6, "full")

        tk.Label(self, text="Data Path (.csv):").grid(row=0)
        tk.Label(self, text="Size:").grid(row=1)
        tk.Label(self, text="Time frame of interest:").grid(row=2)
        tk.Label(self, text="Predict steps:").grid(row=3)
        tk.Label(self, text="Training percentage(%):").grid(row=4)
        tk.Label(self, text="Training cycles for each iteration):").grid(row=5)
        tk.Label(self, text="Plotting(\"Pls\",\"Ls\",\"full\",\"None\"):").grid(row=6)

        def get(e1, e7, e3, e4, e2, e5, e6):
            path = str(e1.get())
            size = int(float(e2.get()))
            increments = int(float(e3.get()))
            steps = int(float(e4.get()))
            train_per = float(e5.get()) / 100
            train_num = int(float(e6.get()))
            plots = str(e7.get())

            on_button(path, plots, increments, steps,

                      size, train_per, train_num)

        bt_pl1 = ttk.Button(self, text='Full',
                            command=lambda: Plotting)

        bt0 = ttk.Button(self, text='Apply',
                         command=lambda: get(e1, e7, e3, e4, e2, e5, e6))

        bt2 = ttk.Button(self, text='Exit',
                         command=self.quit)

        bt0.grid(row=8, column=1)
        bt2.grid(row=8, column=2)

        self.title("Trends - Data Analysis")

        def on_button(path, plots, increments, steps,

                      size, train_per, train_num):
            start = timeit.default_timer()
            self.destroy()

            if path == "sample":
                path = "BTC-USD.csv"
            if path== "example":
                path="example.csv"

            probability_density(path=path, Plots=plots, interest=increments, steps=steps,
                                size=size, training=int(train_per * size), ml_steps=train_num, start=start)

        self.mainloop()
        tk.mainloop()


pop_up_exe()