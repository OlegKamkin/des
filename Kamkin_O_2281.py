from sympy import * # пакет символьных вычислений
import numpy as np # пакет для работы с многомерными массивами
x, y, t, C = symbols("x y t C")
f = x ** 3 - sin(x) / 2 + sqrt(x - 1)
print(f)
print(diff(f, x))
print(integrate(f, x))
print(integrate(f, (x, 2, 5)))
# создаем производную 
dydx = Derivative(y, x) # dydx - имя переменной, обозначающей производную
print(dydx)
eq = Eq(x * dydx + 2, x ** 2)
print(eq)
sol = solve(eq, dydx)[0]
print(eq)
dsol = integrate(sol, x) + C
print(dsol)
# Находим константу интегрирования, решая уравнение y(x0, C) = y0, где x0, y0 - начальное условие
x0, y0 = 1, 2 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0) # подставили C в общее решение
print(dsol0)
# график функции f(x) в интервале от a до b строится командой plot( f(x), (x,a,b) )
p = plot(x ** 2, (x, -1, 1))
print(p)
# график решения dsol0 начальной задачи
p = plot(dsol0, (x, 0.1, 4))
print(p)
# график семейства интегральных кривых общего решения dsol
p = plot(show=False) # пустой график
for c in np.arange(-2, 2, 0.25): # цикл по значениям константы C
  p1 = plot(dsol.subs(C, c), (x, 0.1, 4), show=False) # график новой кривой
  p.extend(p1) # добавляем новый график к общему графику
p.show() # показываем общий график
'''))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))))))))))))'''
x = Symbol('x')
################# 1.1
print('Задание 1.1')
Y_ot_x =2*x**2 - 3*x + 1
print('1.',diff(Y_ot_x))
Y_ot_x = 3*x**2+x-1
print('2.',diff(Y_ot_x))
Y_ot_x = -2*x**2 - x +4
print('3.',diff(Y_ot_x))
Y_ot_x = x**2+4*x+2
print('4.',diff(Y_ot_x))
Y_ot_x = -3*x**2-3*x+4
print('5.',diff(Y_ot_x))
Y_ot_x = 4*x**2-2*x-1
print('6.',diff(Y_ot_x))
Y_ot_x = 5*x**2+x+3
print('7.',diff(Y_ot_x))
Y_ot_x = -x**2-4*x+5
print('8.',diff(Y_ot_x))
Y_ot_x = -4*x**2+4*x-1
print('9.',diff(Y_ot_x))
Y_ot_x = -2*x**2+6*x+5
print('10.',diff(Y_ot_x))
Y_ot_x = -5*x**2+6*x+5
print('11.',diff(Y_ot_x))
Y_ot_x = 4*x**2+2*x+1
print('12.',diff(Y_ot_x))
################# 1.2
print('\nЗадание 1.2')
Y_ot_x = E**(-x)*sin(4*x)
print('1.',diff(Y_ot_x))
Y_ot_x = E**(2*x)*cos(x+1)
print('2.',diff(Y_ot_x))
Y_ot_x = E**(-x)*sqrt(x-2)
print('3.',diff(Y_ot_x))
Y_ot_x = E**(-2*x)*sin(2*x-2)
print('4.',diff(Y_ot_x))
Y_ot_x = E**(2*x)*cos(3*x)
print('5.',diff(Y_ot_x))
Y_ot_x = E**x *sqrt(2*x+1)
print('6.',diff(Y_ot_x))
Y_ot_x = E**(-3*x) *ln(4*x)
print('7.',diff(Y_ot_x))
Y_ot_x = E**(-2*x)*sin(2-x)
print('8.',diff(Y_ot_x))
Y_ot_x = E**(-x)*cos(1-2*x)
print('9.',diff(Y_ot_x))
Y_ot_x = E**x *sqrt(4*x+2)
print('10.',diff(Y_ot_x))
Y_ot_x = E**(2*x) *ln(1-x)
print('11.',diff(Y_ot_x))
Y_ot_x = E**(-2*x) *ln(3+2*x)
print('12.',diff(Y_ot_x))
################# 1.3
print('\nЗадание 1.3')
Y_ot_x = sin(E**(2*t))
print('1.',diff(Y_ot_x))
Y_ot_x = sqrt(sin(3*t))
print('2.',diff(Y_ot_x))
Y_ot_x = E**(cos(1-t))
print('3.',diff(Y_ot_x))
Y_ot_x = ln(1+4*t**3)
print('4.',diff(Y_ot_x))
Y_ot_x = tan(1/(t**2-t)) 
print('5.',diff(Y_ot_x))
Y_ot_x = cos(ln(1-2*t))
print('6.',diff(Y_ot_x))
Y_ot_x = (sin(t**2+t))**3
print('7.',diff(Y_ot_x))
Y_ot_x = asin(sqrt(t))
print('8.',diff(Y_ot_x))
Y_ot_x = 2**(t**3-4*t**2)
print('9.',diff(Y_ot_x))
Y_ot_x = sin(cos(2*t))
print('10.',diff(Y_ot_x))
Y_ot_x = ln(1+E**(-t))
print('11.',diff(Y_ot_x))
Y_ot_x = acos(1-t**2)
print('12.',diff(Y_ot_x))
################# 1.4
print('\nЗадание 1.4')
Y_ot_x = 2*x+3
print('1.',integrate(Y_ot_x))
Y_ot_x = 4-x
print('2.',integrate(Y_ot_x))
Y_ot_x = 3*x-1
print('3.',integrate(Y_ot_x))
Y_ot_x = 2-3*x
print('4.',integrate(Y_ot_x))
Y_ot_x = 5*x+2
print('5.',integrate(Y_ot_x))
Y_ot_x = -2*x
print('6.',integrate(Y_ot_x))
Y_ot_x = 6*x-3
print('7.',integrate(Y_ot_x))
Y_ot_x = 1-4*x
print('8.',integrate(Y_ot_x))
Y_ot_x = 5+2*x
print('9.',integrate(Y_ot_x))
Y_ot_x = 1-4*x
print('10.',integrate(Y_ot_x))
Y_ot_x = -x+5
print('11.',integrate(Y_ot_x))
Y_ot_x = 6-2*x
print('12.',integrate(Y_ot_x))
################# 1.5
dx = Derivative(x)
print('\nЗадание 1.5')
Y_ot_x = E**(6*x-3)
print('1.',integrate(Y_ot_x,(x,0,1)))
Y_ot_x = sin(1-4*x)
print('2.',integrate(Y_ot_x,(x,0, pi)))
Y_ot_x = cos(5+2*x)
print('3.',integrate(Y_ot_x,(x,1,2)))
Y_ot_x = 1/(1-4*x)
print('4.',integrate(Y_ot_x,(x,-1,0)))
Y_ot_x = sqrt(5-x)
print('5.',integrate(Y_ot_x,(x,1,4)))
Y_ot_x = E**(6-2*x)
print('6.',integrate(Y_ot_x,(x,0,3)))
f=cos(2*x+3)
print('7.', integrate(f, (x, 0, 1)))
f=sin(4-x)
print('8.', integrate(f, (x, 0, 4)))
f=1/(3*x-1)
print('9.', integrate(f, (x, 1, 2)))
f=1/(2-3*x)**0.5
print('10.', integrate(f, (x, -1, 0)))
f=E**(5*x+2)
print('11.', integrate(f, (x, 1, 4)))
f=sin(-2*x+4)
print('12.', integrate(f, (x, 0, 2)))
#####1.6
print('Задание 1.6')
dydx = Derivative(y, x)
eq = Eq(x * dydx, 1)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = E, 2 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('1.',dsol0, solve(dsol0-y0,x)[0]==x0)
eq = Eq(dydx-x**2, x)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 1, -1 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('2.',dsol0)
eq = Eq(2*dydx, x+sin(3*x))
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = pi, 0 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('3.',dsol0)
eq = Eq((x+1)*dydx, 2)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 0, -1 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('4.',dsol0)
eq = Eq(dydx+x**3,1)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = -1, 2 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('5.',dsol0, solve(dsol0-y0,x)[0]==x0)
eq = Eq(dydx*x,x**2)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = -1, 2 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('6.',dsol0)
dydx = Derivative(y, x)
eq = Eq(dydx*x**2, 1-x)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 1, 0 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('7.',dsol0)
dydx = Derivative(y, x)
eq = Eq(dydx-cos(x), x+1)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 0, 1 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('8.',dsol0)
dydx = Derivative(y, x)
eq = Eq(dydx+x**3, sin(3*x))
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 0, 0 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('9.',dsol0)
dydx = Derivative(y, x)
eq = Eq(dydx*x, sqrt(x)-x**2)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 4, 1 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('10.',dsol0, solve(dsol0-y0,x)[0]==x0)
dydx = Derivative(y, x)
eq = Eq(dydx-3*x**2, cos(x))
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = pi, 0 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('11.',dsol0)
dydx = Derivative(y, x)
eq = Eq(sqrt(x)*dydx, 1+2*x**2)
sol = solve(eq, dydx)[0]
dsol = integrate(sol, x) + C
x0, y0 = 9, -1 # см. условие задачи выше
X = dsol.subs(x, x0) # подставили x0 вместо x в решение dsol
eq = Eq(y0, X) # приравняли к y0
C0 = solve(eq)[0] # нашли С
dsol0 = dsol.subs(C, C0)
print('12.',dsol0, solve(dsol0-y0,x)[0]==x0)













