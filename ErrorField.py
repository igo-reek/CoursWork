import copy
import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def schedule(self, *args, **kwargs):
        fig = plt.figure(figsize=(16, 14))
        ax = plt.gca()
        xlist = []
        ylist = []
        xlim = 310
        ylim = 310
        if "title" in kwargs.keys():
            ax.set_title("%s" % (kwargs.pop('title')), fontsize=23)
        if "xlabel" in kwargs.keys():
            ax.set_xlabel("%s" % (kwargs.pop('xlabel')), fontsize=15)
        if "ylabel" in kwargs.keys():
            ax.set_ylabel("%s" % (kwargs.pop('ylabel')), fontsize=15)
        if "xlim" in kwargs.keys():
            xlim = kwargs.pop('xlim')
        if "ylim" in kwargs.keys():
            ylim = kwargs.pop('ylim')
        if "label" in kwargs.keys():
            label = (kwargs.pop('label'))
            # print(label)
        if "circle" in kwargs.keys():
            if kwargs.pop('circle'):
                for i, arg in enumerate(args):
                    Error = arg
                    lab = label[i]
                for i in range(len(self.Rx)):
                    ax.add_patch(plt.Circle((self.Rx[i], self.Ry[i]), Error[i], color='r'))
                ax.add_patch(plt.Circle((self.Rx[0], self.Ry[0]), Error[0], color='r', label=u'%s' % lab))
            else:
                for i, arg in enumerate(args):
                    if i % 2 == 0:
                        xlist.append(arg)
                    else:
                        ylist.append(arg)
            for x, y, lab in zip(xlist, ylist, label):
                ax.plot(x, y, label=u'%s' % lab)
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        ax.grid(True)
        ax.legend(loc=1, prop={'size': 15})
        plt.show()


class Mistakes(Plot):

    def __init__(self, Rx, Ry, Pelengx, Pelengy, d, base, sigma_Rp, sigma_alfa):
        self.Pelengy = Pelengy
        self.Pelengx = Pelengx
        self.Ry = Ry
        self.Rx = Rx
        self.D = d
        self.base = base
        self.sigma_Rp = sigma_Rp
        self.sigma_alfa = sigma_alfa

    def straight_line(self):  # direct coefficients
        A = []
        B = []
        C = []
        for i in range(len(self.Rx)):
            if self.Rx[i] < self.base:
                p = 0
            else:
                p = 1
            A = np.append(A, self.Pelengy[p] - self.Ry[i])
            B = np.append(B, self.Rx[i] - self.Pelengx[p])
            C = np.append(C, self.Pelengx[p] * self.Ry[i] - self.Rx[i] * self.Pelengy[p])
        return [A, B, C]

    def direct(self, coefficient):
        for i in range(len(self.Rx)):
            if self.Rx[i] < self.base:
                p = 0
            else:
                p = 1
            x = np.linspace(self.Pelengx[p], self.Rx[i], 400)
            y = np.linspace(self.Pelengy[p], self.Ry[i], 400)
            x, y = np.meshgrid(x, y)
            plt.contour(x, y, (coefficient[0][i] * x + coefficient[1][i] * y + coefficient[2][i]), [1])  # direct
        plt.grid(True)
        plt.xlim(0, 300)
        plt.ylim(0, 300)
        plt.show()

    def width_difference(self):  # distance difference
        q = []
        for i in range(len(self.Rx)):
            b = (self.Rx[i] - self.base) ** 2 + self.Rx[i] ** 2 + (self.D / 2) ** 2
            alfa = b / 2 - (np.sqrt(b ** 2 - 4 * ((self.Rx[i] - self.base) ** 2) * (self.D / 2) ** 2)) / 2
            q = np.append(q, np.sqrt(abs(alfa)))
        return q

    def Hyperbola(self, oy, q, num_r):
        yy = []
        for ii in range(len(oy)):
            if self.Rx[num_r] < self.base:
                yy = np.append(yy,
                               self.base - np.sqrt(
                                   (oy[ii] ** 2 / ((self.D / 2) ** 2 - (q[num_r]) ** 2) + 1) * (q[num_r]) ** 2))
            else:
                yy = np.append(yy,
                               self.base + np.sqrt(
                                   (oy[ii] ** 2 / ((self.D / 2) ** 2 - (q[num_r]) ** 2) + 1) * (q[num_r]) ** 2))
        return yy

    def plot_Hyperbola(self, oy):
        XX = []
        for i in range(len(self.Rx)):
            Hyp = self.Hyperbola(oy, self.width_difference(), i)
            XX = np.append(XX, Hyp)
        X = [XX[i * len(oy):i * len(oy) + len(oy)] for i in range(len(self.Rx))]
        return X

    def deriv(self, x, Num_r, width):
        h = 0.001  # step-size
        return (self.f1(x + h, Num_r, width) - self.f1(x, Num_r, width)) / h  # definition of derivative

    def tangent_line(self, Num_r, x_0, a, b, width):
        x = np.linspace(a, b, 100)
        y_0 = self.f1(x_0, Num_r, width)
        y_tan = self.deriv(x_0, Num_r, width) * (x - x_0) + y_0
        # print(y)
        return y_tan

    def f1(self, x, Num_r, width):
        d = 100
        q = width
        return np.sqrt((((self.base - x) ** 2 / (q[Num_r]) ** 2) - 1) * ((d / 2) ** 2 - (q[Num_r]) ** 2))

    y_tangent_hyp = []
    x_tangent_hyp = []
    xtan = []
    x_peleng = []
    y_peleng = []

    def plot(self, DirectCoefficients, width):
        y_tangent_hyp = []
        x_tangent_hyp = []
        x_peleng = []
        y_peleng = []
        for Num_r in range(len(self.Rx)):
            a = self.Rx[Num_r] + 1  # boundaries of integration
            b = self.Rx[Num_r] - 1
            tang = self.tangent_line(Num_r, self.Rx[Num_r], a, b, width)
            xtan = np.linspace(a, b, 100)
            y_tangent_hyp = np.append(y_tangent_hyp, tang)
            x_tangent_hyp = np.append(x_tangent_hyp, xtan)
            if self.Rx[Num_r] < self.base:
                p = 0
            else:
                p = 1
            xpel = np.linspace(self.Pelengx[p], self.Rx[Num_r], 100)
            x_peleng = np.append(x_peleng, xpel)
            y_peleng = np.append(y_peleng, (-DirectCoefficients[0][Num_r] * xpel - DirectCoefficients[2][Num_r]) /
                                 DirectCoefficients[1][Num_r])

        y_tang_hyp = [y_tangent_hyp[i * 100:i * 100 + 100] for i in range(len(self.Rx))]
        x_tang_hyp = [x_tangent_hyp[i * 100:i * 100 + 100] for i in range(len(self.Rx))]
        x_pel = [x_peleng[i * 100:i * 100 + 100] for i in range(len(self.Rx))]
        y_pel = [y_peleng[i * 100:i * 100 + 100] for i in range(len(self.Rx))]
        return x_pel, y_pel, y_tang_hyp, x_tang_hyp

    def alfa_psi_R(self, x_peleng, y_peleng, x_hyp, y_hyp, Pelengx):
        x11 = []
        x12 = []
        x21 = []
        x22 = []
        y11 = []
        y12 = []
        y21 = []
        y22 = []
        xpel = []
        ypel = []
        xhyp = []
        yhyp = []
        alfa = []
        psi = []
        R = []
        for i in range(len(x_hyp)):
            x_11 = x_peleng[i][0]
            x11 = np.append(x11, x_11)
            y_11 = y_peleng[i][0]
            y11 = np.append(y11, y_11)
            x_12 = x_peleng[i][len(x_peleng[i]) - 1]
            x12 = np.append(x12, x_12)
            y_12 = y_peleng[i][len(y_peleng[i]) - 1]
            y12 = np.append(y12, y_12)
            x_21 = x_hyp[i][0]
            x21 = np.append(x21, x_21)
            y_21 = y_hyp[i][0]
            y21 = np.append(y21, y_21)
            x_22 = x_hyp[i][len(x_hyp[i]) - 1]
            x22 = np.append(x22, x_22)
            y_22 = y_hyp[i][len(y_hyp[i]) - 1]
            y22 = np.append(y22, y_22)
            xpel = np.append(xpel, x12[i] - x11[i])
            ypel = np.append(ypel, y12[i] - y11[i])
            xhyp = np.append(xhyp, x21[i] - x22[i])
            yhyp = np.append(yhyp, y21[i] - y22[i])
            alfa = np.append(alfa, abs(np.arctan(xhyp[i] / yhyp[i]) - np.arctan(xpel[i] / ypel[i])) * 180 / np.pi)
            if self.Rx[i] < 0:
                ps = abs(np.arctan(xhyp[i] / yhyp[i]) - np.arctan(xpel[i] - Pelengx[1] / ypel[i])) * 180 / np.pi
                psi = np.append(psi, ps)
            else:
                ps = abs(np.arctan(xhyp[i] / yhyp[i]) - np.arctan(xpel[i] - Pelengx[0] / ypel[i])) * 180 / np.pi
                psi = np.append(psi, ps)
            r = np.sqrt(xpel[i] ** 2 + ypel[i] ** 2)
            R = np.append(R, r)
        return [alfa, psi, R]

    def sigma(self, s_Rp, s_alfa, angle):
        sig = []
        for i in range(len(self.Rx)):
            sig = np.append(sig, np.sqrt(
                (angle[2][i] * s_alfa) ** 2 + (s_Rp / 2 / np.sin(angle[1][i] * np.pi / 180)) ** 2) / np.sin(
                (angle[0][i] * np.pi / 180)))
        return sig

    def func(self):
        self.straight_line()
        DirectCoefficients = self.straight_line()
        # self.direct(DirectCoefficients)
        oy = np.linspace(0, 300, 300)
        self.plot_Hyperbola(oy)
        width = self.width_difference()
        x_pel, y_pel, x_tang_hyp, y_tang_hyp = self.plot(DirectCoefficients, width)
        alfa_psi_R = self.alfa_psi_R(x_pel, y_pel, x_tang_hyp, y_tang_hyp, self.Pelengx)
        ErrorRadius = self.sigma(self.sigma_Rp, self.sigma_alfa, alfa_psi_R)
        return ErrorRadius


class Kalman(Mistakes):
    """

    для построения изолиний на поле ошибок

    def show_shape(self, ErrorRadius):
        ErrorRadiusSort = ErrorRadius.argsort()

        return [ErrorRadiusSort[i:i + int(len(ErrorRadiusSort) / 7)] for i in
                range(0, len(ErrorRadiusSort), int(len(ErrorRadiusSort) / 7))]
    """

    def function(self, ErrorRadius):
        Traektoria, R = self.noize(ErrorRadius)
        self.schedule(Traektoria[0], Traektoria[1], R[0], R[1],
                      circle=False,
                      title="Координаты ЛА до обработки фильтром Калмана",
                      label=['зашумленная траектория', 'истинная траектория'],
                      xlabel='km',
                      ylabel='km')
        Kalmancoordinates = self.Kalmanfilter(Traektoria[0], Traektoria[1], ErrorRadius)
        self.schedule(Kalmancoordinates[0], Kalmancoordinates[1], Traektoria[0], Traektoria[1],
                      circle=False,
                      title="Координаты ЛА после обработки фильтром Калмана",
                      label=['отфильтрованная траектория', 'зашумленная траектория'],
                      xlabel='km',
                      ylabel='km')
        dev, devkalman = self.standard_deviation(self.Rx, ErrorRadius)
        self.schedule(dev[0], dev[1], devkalman[0], devkalman[1],
                      circle=False,
                      title='СКО для выборки из 200 траекторий',
                      label=['СКО до обработки', 'СКО после обработки'],
                      xlabel='Порядковый номер измерения координаты',
                      ylabel='km',
                      xlim=len(dev[1]),
                      ylim=max(dev[1]))

    def noize(self, ErrorRadius):
        traektoria = [[], []]
        for i in range(len(self.Rx)):
            x_rand = np.random.normal(self.Rx[i], ErrorRadius[i])
            y_rand = np.random.normal(self.Ry[i], ErrorRadius[i])
            traektoria[0] = np.append(traektoria[0], x_rand)
            traektoria[1] = np.append(traektoria[1], y_rand)
        return [traektoria[0], traektoria[1]], [self.Rx, self.Ry]

    def Kalmanfilter(self, traekx, traeky, ErrorRadius):
        b = 0.6
        a = [0, 0.25, 0.5, 1, 1.5, 1.75, 2, 2.5, 3.5, 4, 8]
        ErrorRadius_index = np.zeros(len(ErrorRadius))
        x = np.linspace(0, 8, 11)
        for i in range(10):
            for q in range(len(ErrorRadius)):
                if a[i] < ErrorRadius[q] < a[i + 1]:
                    ErrorRadius_index[q] = b
            b = b - 0.05
        beta = []
        l = len(traekx)
        xk = np.zeros(l)
        yk = np.zeros(l)
        uk_x = np.zeros(l)
        uk_y = np.zeros(l)
        xk[0] = traekx[0]
        yk[0] = traeky[0]
        uk_x[0] = traekx[1] - traekx[0]
        uk_y[0] = traeky[1] - traeky[0]
        t = 1
        alfa = copy.copy(ErrorRadius_index)
        for alf in alfa:
            beta.append(alf ** 2 / (2 - alf))
        for i in range(1, l):
            " X component"
            xk[i] = xk[i - 1] + uk_x[i - 1] * 1 + alfa[i] \
                    * (traekx[i] - (xk[i - 1] + uk_x[i - 1] * t))
            uk_x[i] = beta[i] / t * (traekx[i] - (xk[i - 1] + uk_x[i - 1] * t))
            "Y component"
            yk[i] = yk[i - 1] + uk_y[i - 1] * 1 + alfa[i] \
                    * (traeky[i] - (yk[i - 1] + uk_y[i - 1] * t))
            uk_y[i] = beta[i] / t * (traeky[i] - (yk[i - 1] + uk_y[i - 1] * t))
        return xk, yk

    def standard_deviation(self, Rx, ErrorRadius):
        traektorx = [[np.random.normal(Rx[i], ErrorRadius[i]) for i in range(len(Rx))] for q in range(200)]
        traektory = [[np.random.normal(Rx[i], ErrorRadius[i]) for i in range(len(Rx))] for q in range(200)]

        numiterx = [[traektorx[q][i] for q in range(200)] for i in range(len(Rx))]
        numitery = [[traektory[q][i] for q in range(200)] for i in range(len(Rx))]
        print(len(numitery))

        stdx = []  # standard deviation noize traektoria
        stdy = []
        stdxkalmsum = []  # standard deviation kalman traektoria
        stdykalmsum = []
        xkalmsum = []
        ykalmsum = []

        for i in range(len(numiterx)):
            stdx = np.append(stdx, np.std(numiterx[i]))
            stdy = np.append(stdy, np.std(numitery[i]))

        for i in range(200):
            xkalm, ykalm = self.Kalmanfilter(traektorx[i], traektory[i], ErrorRadius)
            xkalmsum = xkalmsum + [xkalm]
            ykalmsum = ykalmsum + [ykalm]

        numxkalmsum = [[xkalmsum[q][i] for q in range(200)] for i in range(len(Rx))]
        numykalmsum = [[ykalmsum[q][i] for q in range(200)] for i in range(len(Rx))]

        for i in range(len(numiterx)):
            stdxkalmsum = np.append(stdxkalmsum, np.std(numxkalmsum[i]))
            stdykalmsum = np.append(stdykalmsum, np.std(numykalmsum[i]))

        # standard deviation x and y
        stdxy = []
        stdxykalmsum = []
        for i in range(len(stdx)):
            stdxy.append(np.sqrt(stdx[i] ** 2 + stdy[i] ** 2))
            stdxykalmsum.append(np.sqrt(stdxkalmsum[i] ** 2 + stdykalmsum[i] ** 2))
        return [range(len(stdxy)), stdxy], [range(len(stdxykalmsum)), stdxykalmsum]