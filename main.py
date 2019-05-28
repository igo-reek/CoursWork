import ErrorField
import numpy as np
Rx = [i * 10 + 15 for i in range(0, 30)] * 26
Ry = [q * 10 + 4 for q in range(4, 30) for i in range(0, 30)]
# RxLine = [(i * 10 + 5) / 4 for i in range(0 * 2, 60 * 2)]
# RyLine = [(-i * 5 + 5) / 4 for i in range(-90 * 2, -30 * 2)]
# RxLine = np.linspace(70, 100, 100)
# RyLine = RxLine**2/80+50
RxLine = np.linspace(100, 200, 100)
RyLine = RxLine**2/160 - 3.2*RxLine + 470
print(RyLine)
Pelengx = [140, 160]
Pelengy = [0, 0]
d = Pelengx[1] - Pelengx[0]
base = (Pelengx[0] + Pelengx[1]) / 2
sigma_Rp = 0.001
sigma_alfa = 0.007
Rx1 = [1, 2]
Ry1 = [1, 5]
Field = ErrorField.Kalman(Rx, Ry, Pelengx, Pelengy, d, base, sigma_Rp, sigma_alfa)
ErrorRadiusField = Field.func()
Field.schedule(ErrorRadiusField,
               circle=True,
               title="Поле ошибок",
               label=['величина ошибки'],
               xlabel='km',
               ylabel='km')
Line = ErrorField.Kalman(RxLine, RyLine, Pelengx, Pelengy, d, base, sigma_Rp, sigma_alfa)
ErrorRadiusLine = Line.func()
Line.schedule(ErrorRadiusLine,
              circle=True,
              title="Поле ошибок для прямолинейной траектории",
              label=['величина ошибки'],
              xlabel='km',
              ylabel='km')
Line.function(ErrorRadiusLine)


