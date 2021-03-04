import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from fresnaq_functions import fresnaq_functions
from scipy.special import erfc

fresnaq = fresnaq_functions()

def test_starshadequad():
    verb = True
    Aquad = ['g','a','u'][0]
    Np = 16
    A = lambda t: 1+0*t     #First test disc radius r0 (filled petals)
    r0, r1 = 0.7, 1.3
    n, m = 20, 30

    xj,yj,wj, bx, by = fresnaq.starshadequad(Np, A, r0, r1, n,m,Aquad=Aquad,verb=verb)
    derr = wj.sum() - np.pi*r1**2
    print(f'disc err: {derr:.2e}\n')
    plt.figure()
    plt.scatter(xj,yj,s=2,c=wj)

    designs = ['erf', 'NI2']
    for des in designs:

        if des == 'erf':
            Np, r0, r1 = 16, 7, 14
            beta = 3
            A = lambda r: erfc(beta*(2*r-(r0+r1))/(r1-r0))/2
            ms = np.arange(30,50+10,10)
        elif des == 'NI2':

            breakpoint()

        xj, yj, wj, bx, by = fresnaq.starshadequad(Np, A, r0, r1, n, ms[0])

        for mm in ms:
            xj, yj, wj, bx, by = fresnaq.starshadequad(Np, A, r0, r1, n, mm)
            print(wj.sum())


        plt.figure()
        plt.scatter(xj,yj,s=2,c=wj)
        for i in range(bx.shape[1]):
            plt.plot(bx[:,i], by[:,i], 'x')
            # breakpoint()

        breakpoint()


if __name__ == '__main__':

    test_starshadequad()
