# -*- coding: utf-8 -*-
"""
v1 - Created on Tue Sep  1 07:54:17 2020
@author: Marcela
v2 - Created on Mon Apr  1 17:34:27 2024
@author: mumor
"""
# 1GDL 
import numpy as np
from math import sin
from math import cos
from math import exp, sqrt, atan2
from math import pi
from scipy.integrate import odeint
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

import csv

# parametros
m = 30                           # kg
k = 15000                        # N/m
c= 0.0001
c= 2*0.02*m*(np.sqrt(k/m))
wn = np.sqrt(k/m)                # frequencia natural
xi = (c/m)/(2*wn)
wd = wn*sqrt(1-xi**2)

delta = 0.01
time = np.arange(0.0, 10.0, delta)   # time [s]
#t = np.linspace(0,3,100, endpoint=True)
Tp = 2*pi/wn                     # período

# Obtenção do descolamento e velocidade por solucao numerica odetin
# Equacao do sistema
def sistemVibL(y,t):
    xp = y[1]
    xpp = -k/m * y[0] -c/m * y[1] #+ F/m
    dy = [xp, xpp]
    return dy

# Definindo uma condicao initial

ui = 1.0
vi = 0.0

A = 1                            # Amplitude
phi = 0                          # fase
A = sqrt(ui**2 + ((vi+ui*xi*wn)/wd)**2)
phi = atan2(((vi+ui*xi*wn)/wd),wn)
#print(A)
#print(phi)

# Solucionando o problema com um o odeint
# Using odeint function -> []=odeint(equations, [initial conditions], time)
uvnum = odeint(sistemVibL, [ui,vi], time)

# deslocamento
u = [ A*exp(-xi*wn*t)*cos(wd*t-phi) for t in time] # eq.3.35 livro Aline
v = [-A*exp(-xi*wn*t)*((-xi*wn)*cos(wd*t-phi)+(wd)*sin(wd*t-phi)) for t in time]           # derivado da eq.3.35
# print(v)


# Plotando os resultados
plt.figure(1)
plt.plot(time,u)
plt.plot(time,uvnum[:,0])
plt.xlabel('Time(s)')
plt.ylabel('Deslocamento(m)')
plt.show()

plt.figure(2)
plt.plot(time,v)
plt.plot(time,uvnum[:,1])
plt.xlabel('Time(s)')
plt.ylabel('Velocidade(m/s)')
plt.show()

# plt.figure(3)
# plt.plot(time,u)
# plt.plot(time,uvnum[:,0])
# plt.xlabel('Time(s)')
# plt.ylabel('Velocidade(m/s)')
# plt.xlim(95,100)
# plt.ylim(-0.05,0.05)
# plt.show()

# plt.figure(4)
# plt.plot(time,v)
# plt.plot(time,uvnum[:,1])
# plt.xlabel('Time(s)')
# plt.ylabel('Velocidade(m/s)')
# plt.xlim(95,100)
# plt.ylim(-1.0,1.0)
# plt.show()


# Análise de frequência uma condicao initial

u_frq = np.fft.fft(uvnum[:,0])
n = len(uvnum[:,0])
nhalf = int(np.floor(n/2))
frq = np.fft.fftfreq(n, d=delta)

plt.figure(5)
plt.semilogy(frq[0:nhalf-1],np.abs(u_frq[0:nhalf-1]))
plt.xlabel('Frequence (Hz)')
plt.ylabel('abs(U) (m/s)')
plt.xlim(0,120)


u_frq_max = np.max(abs(u_frq))
peaks_index, properties = find_peaks(np.abs(u_frq),height = u_frq_max/2)
print('indice pico : ',peaks_index[0])
print('freq. pico',frq[peaks_index[0]])
print('freq. teo:',wn/(2*np.pi))

plt.semilogy(frq[peaks_index[0]],np.abs(u_frq[peaks_index[0]]),'ro')
plt.show()

##########
# gravar dados em arquivo CSV
##########
csv_file = '1gdl_ex_4.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file,delimiter=';')
    writer.writerows( np.column_stack(("Time","U_t","V_t")))
    writer.writerows( np.column_stack((time,uvnum[:,0],uvnum[:,1])) )

# Print a success message
print(f"Data successfully written to {csv_file}")

