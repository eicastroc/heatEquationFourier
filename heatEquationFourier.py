import marimo

__generated_with = "0.1.43"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import functools
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import bisect
    return bisect, functools, np, plt


@app.cell
def __(mo):
    mo.md(
    """
    # Solución analítica de la ecuación de conducción de calor en régimen transitorio

    **Suposiciones:**

    - Conducción de calor 1D a través de una placa.

    - Área transversal de la sección es constante.

    - Difusividad térmica constante.

    - No hay generación de calor.

    """    
    )
    # Initial source for notebook idea:   https://youtu.be/7GeQJhF5B7E?si=T3sVbjsIPmshFNH1
    return


@app.cell
def __(mo):
    mo.md(
    r"""
    ## Ecuación de conducción en 1D régimen transitorio.

    $$\frac{1}{\alpha} \frac{\partial T}{\partial t} = \frac{\partial^2 T}{\partial x^2}$$

    Donde:

    - $\alpha$: difusividad térmica, en $[m^2s^{-1}]$.
    - $T$: Temperatura, en $[K]$.
    - $t$: tiempo, en $[s]$.
    - $x$: posición, en $[m]$.

    """
    )
    return


@app.cell
def __(mo):
    mo.md(
    r"""

    **La solución de la ecuación consiste en encontrar $T(x, t)$ sujeto a las siguientes condiciones:**

    ### Condiciones iniciales

    El perfil de temperatura inicial es de temperatura constante, $T_i$, en todo el dominio.

    $$T(x, 0) = T_i$$

    ### Condiciones de frontera

    Condición de simetría en $x=0$.

    $$\left. \frac{\partial T}{\partial x} \right|_{x=0} = 0$$

    Condición de Cauchy en $x=L$:

    $$- \lambda \left. \frac{\partial T}{\partial x} \right|_{x=L} = h \left( T(L, t) - T_{\infty} \right)$$

    """
    )
    return


@app.cell
def __(mo):
    mo.md(
    r"""
    ## Ecuación adimensional

    Para obtener la solución analítica mediante series de Fourier, la ecuación se escribe de forma adimensional.

    $$\frac{\partial \theta^*}{\partial \mathrm{Fo}} = \frac{\partial^2 \theta^*}{\partial {x^*}^2}$$

    La ecuación caracteriza el cambio de la temperatura adimensional, $\theta$, en función de tres parámetros:

    $$\theta^* = \theta^* \left( x^*, \mathrm{Fo}, \mathrm{Bi} \right)$$



    """
    )
    return


@app.cell
def __(mo):
    mo.md(
    r"""

    ### Temperatura adimensional

    Para el problema planteado, la temperatura adimensional está limitada en el rango $[0, 1]$.

    $$\theta^* = \frac{\theta}{\theta_i} = \frac{T - T_{\infty}}{T_i - T_{\infty}}$$

    ### Posición adimensional

    Para el problema planteado, la posición adimensional está limitada en el rango $[0, 1]$.

    $$x^* = \frac{x}{L}$$

    ### Número de Fourier

    El número de Fourier representa la relación entre la tasa de almacenamiento de la energía calorífica, y la tasa de conducción de calor.

    $$\mathrm{Fo} = \frac{\alpha t}{L^2}$$

    ### Número de Biot

    El número de Biot representa la relación entre la tasa de transferencia de calor por convección del fluido y la tasa de transferencia de calor por conducción en el cuerpo.

    $$\mathrm{Bi} = \frac{h \,L}{\kappa}$$

    """
    )
    return


@app.cell
def __(mo):
    mo.md(
    r"""
    ## Solución analítica por series de Fourier

    Al efectuar la integración de la ecuación mediante el método de separación de variables, se obtiene la ecuación que dá el perfil adimensional de temperatura en la placa.

    $$\theta^* = \frac{T - T_{\infty}}{T_i - T_{\infty}} = \sum_{n=1}^{\infty} A_n \exp \left( -\xi_n^2 \mathrm{Fo} \right) \cos \left( \xi_n x^* \right)$$

    $$A_n = \frac{2 \sin ({\xi_n})}{\xi_n + \sin(\xi_n) \cos(\xi_n)}$$



    En esta ecuación, los términos $\xi_n$ en la suma infinita, son las raíces positivas de la ecuación trascendental $\xi \tan(\xi) = \mathrm{Bi}$. 
    """
    )
    # $$A_n = \frac{4 \sin ({\xi_n})}{2 \xi_n + \sin (2 \xi_n)}$$
    return


@app.cell
def __(mo):
    mo.md(
    r"""
    ## Comportamiento de la función trascendental

    $$f(\xi) = \xi \tan (\xi) - \mathrm{Bi}$$

    ### Descripción gráfica

    La función trascendental **tiene simetría con respecto al eje vertical** (función par), y **es periódica**.

    """    
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Número de Biot $(\mathrm{Bi})$:
        """
    )
    return


@app.cell
def __(mo, show_element):
    Bi0_value = mo.ui.number(start=-25, stop=20, step=1, value=0)
    show_element(Bi0_value)
    return Bi0_value,


@app.cell
def __(Bi0_value, np, plt, xitanxi_Bi):
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)

    # Horizontal and vertical lines in the graph
    ax0.axhline(0, ls='--', color='k')
    ax0.axvline(0, ls='--', color='k')

    # Equal convection and conduction: Bi = 1
    _xi = np.linspace(-3*np.pi, 3*np.pi, 1000)
    _tanxi_bi = xitanxi_Bi(_xi, Bi0_value.value)
    _BiValue = Bi0_value.value
    _label = "".join([r"$\mathrm{Bi} = $", str(_BiValue)])
    ax0.plot(_xi, _tanxi_bi, 
            ls='', marker='.', color='k', label=_label)
    ax0.plot(_xi, _tanxi_bi, 
            ls='-', color='k', alpha=0.5)

    # Format graph
    ax0.set_xlim(-3*np.pi, 3*np.pi)
    ax0.set_ylim(-5, 5)

    ax0.set_xlabel(r"$\xi$")
    ax0.set_ylabel(r"$\xi \, \tan (\xi) - \mathrm{Bi}$")

    ax0.set_xticks([-3*np.pi, -2.5*np.pi, -2*np.pi, -1.5*np.pi, -1*np.pi, -0.5*np.pi,
                   0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 2.5*np.pi, 3*np.pi])
    ax0.set_xticklabels([r"${-3 \pi}$", r"$-\frac{5\pi}{2}$", 
                         r"${-2 \pi}$", r"$-\frac{3\pi}{2}$",
                         r"${-\pi}$", r"$-\frac{\pi}{2}$",
                        r"$0$", r"$\frac{\pi}{2}$", r"${\pi}$", 
                        r"$\frac{3\pi}{2}$", r"${2 \pi}$",
                        r"$\frac{5\pi}{2}$", r"${3 \pi}$"])
    ax0.legend(loc='upper right')

    ax0.grid(ls='--', color='lightgray')

    plt.gca()
    return ax0, fig0


@app.cell
def __(mo):
    mo.md(
    r"""
    El rango de interés es aquel en que contiene las raíces positivas, cuando el número de Biot es positivo; por lo que se ha graficado únicamente la función para valores de $\xi \ge 0$.

    Se considera el caso en que $\mathrm{Bi}=1$, y los casos límite en que $\mathrm{Bi} \rightarrow 0$, y $\mathrm{Bi} \rightarrow \infty$.

    - En el límite cuando $\mathrm{Bi} \rightarrow 0$, las raíces están dadas por $n \pi$. 

    - En el límite cuando $\mathrm{Bi} \rightarrow \infty$, las raíces están dadas por $n \pi + \frac{1}{2} \pi$.

    Donde , $n \ge 0$, es un número natural.


    """    
    )
    return


@app.cell
def __(np, plt, xitanxi_Bi):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Horizontal and vertical lines in the graph
    ax.axhline(0, ls='--', color='k')


    # Limiting case: Bi --> 0
    _xi = np.linspace(0, 3*np.pi, 1000)
    _tanxi_bi = xitanxi_Bi(_xi, 0.01)
    label = r"$\mathrm{Bi} \rightarrow 0$"
    ax.plot(_xi, _tanxi_bi, 
            ls='-', color='C0', label=label)


    # Equal convection and conduction: Bi = 1
    _xi = np.linspace(0, 3*np.pi, 1000)
    _tanxi_bi = xitanxi_Bi(_xi, 1)
    label = r"$\mathrm{Bi} = 1$"
    ax.plot(_xi, _tanxi_bi, 
            ls='-', color='C1', label=label)

    # Limiting case: Bi --> \infty.
    _xi = np.linspace(0.5*np.pi - 1e-15, 0.5*np.pi + 1e-15, 100)
    _tanxi_bi = xitanxi_Bi(_xi, 100)
    label = r"$\mathrm{Bi} \rightarrow \infty$"
    ax.plot(_xi, _tanxi_bi, 
            ls='-', color='C2', label=label)
    _xi = np.linspace(1.5*np.pi - 1e-15, 1.5*np.pi + 1e-15, 100)
    _tanxi_bi = xitanxi_Bi(_xi, 100)
    ax.plot(_xi, _tanxi_bi, ls='-', color='C2')
    _xi = np.linspace(2.5*np.pi - 1e-15, 2.5*np.pi + 1e-15, 100)
    _tanxi_bi = xitanxi_Bi(_xi, 100)
    ax.plot(_xi, _tanxi_bi, ls='-', color='C2')

    # Format graph
    ax.set_xlim(0, 3*np.pi)
    ax.set_ylim(-5, 5)

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\xi \, \tan (\xi) - \mathrm{Bi}$")

    ax.set_xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 2.5*np.pi, 3*np.pi])
    ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"${\pi}$", 
                        r"$\frac{3\pi}{2}$", r"${2 \pi}$",
                        r"$\frac{5\pi}{2}$", r"${3 \pi}$"])
    ax.legend(loc='upper right')

    ax.grid(ls='--', color='lightgray')

    plt.gca()
    return ax, fig, label


@app.cell
def __(mo):
    mo.md(
    r"""

    ### Estrategia para encontrar las raíces mediante un método numérico

    Para diseñar una estrategia para encontrar las raíces con un algoritmo robusto, se puede considerar lo siguiente:

    - La primera raíz positiva se encuentra dentro del rango $[0, \frac{\pi}{2}]$.

    - Las otras raíces positivas estan contenidas dentro del rango periódico $\left[m \pi - \frac{1}{2} \pi,  m \pi + \frac{1}{2} \pi\right]$.

    Donde, $m \ge 1$, es un número natural.

    El algoritmo que se utiliza en la aplicación numérica presentada en este documento, utiliza el método de bisección, que requiere que el producto $f(\xi_1) \, f(\xi_2) < 0$ para obtener convergencia. Para cada raíz buscada, se debe considerar la periodicidad de la función para determinar los valores iniciales a utilizar en el método de bisección.
    #
    """    
    )
    return


@app.cell
def __(mo):
    mo.md(
    r"""
    # Aplicación numérica

    La aplicación permite estudiar el efecto que tienen sobre el perfil de temperatura adimensional, $\theta^*$; el número de Fourier, $\mathrm{Fo}$, el número de Biot, $\mathrm{Bi}$, y el número de raíces que se consideren para el cálculo de la solución.
    """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        Número de Fourier:
        """
    )
    return


@app.cell
def __(mo, show_element):
    Fo_value = mo.ui.number(start=0, stop=10, step=0.005, value=0)
    show_element(Fo_value)
    return Fo_value,


@app.cell
def __(mo):
    mo.md(
        """
        Número de Biot:
        """
    )
    return


@app.cell
def __(mo, show_element):
    Bi_value = mo.ui.number(start=0.0, stop=1000, step=1, value=1)
    show_element(Bi_value)
    return Bi_value,


@app.cell
def __(mo):
    mo.md(
        """
        Número de raíces en la solución
        """
    )
    return


@app.cell
def __(mo, show_element):
    Nroots_value = mo.ui.number(start=1, stop=1000, step=1, value=20)
    show_element(Nroots_value)
    return Nroots_value,


@app.cell
def __(Bi_value, Fo_value, Nroots_value, dimlessT, find_eigen, np, plt):
    def _plotT(_Fo, _Bi, _eigs):
        # Dimensionless position, [0, 1]
        _xAdim = np.linspace(0, 1, 1001)
        # Dimensionless temperature profile
        _TAdim = dimlessT(_xAdim, _Fo, _eigs)
        # Plot axis of symmetry
        plt.axvline(x=0, ls='--', color='k')
        # Plot T profile around the symmetry axis
        plt.plot(_xAdim, _TAdim, ls = '-', color='k')
        plt.plot(np.flip(-_xAdim), np.flip(_TAdim), ls='-', color='k')
        # Format the graph
        plt.grid(ls='--', color='lightgray')
        plt.xlim(-1, 1)
        plt.ylim(0, 1.5)
        plt.xlabel(r"Posicion adimensional, $x^*$")
        plt.ylabel(r"Temperatura adimensional, $\theta^*$")
        return plt.gca()


    # Fourier number
    _Fo = Fo_value.value

    # Eigenvalues for given Bi and number of roots
    _Bi = Bi_value.value
    _nRoots = int(Nroots_value.value)
    _eigs = find_eigen(_Bi, _nRoots)



    # Plot (showing the symmetry of the problem)
    _plotT(_Fo, _Bi, _eigs)

    return


@app.cell
def __(mo):
    def show_element(element):
        if element is not None:
            return mo.hstack([element], "center")
    return show_element,


@app.cell
def __(np):
    def xitanxi_Bi(xi, Bi):
        """
        Function returning Eigenvalues for the solution of 1D Fourier Equation
        Eigenvalues are found at all the roots of this equation.

        Parameters
        ----------
        xi : FLOAT
            Eigenvalues.
        Bi : FLOAT
            Biot number, Bi=hL/k.

        Returns
        -------
        FLOAT
            The value of the equation for which.

        """
        return xi*np.tan(xi) - Bi

    return xitanxi_Bi,


@app.cell
def __(bisect, np, xitanxi_Bi):
    def find_eigen(Bi, nEigen, eps=1e-8):
        """
        Algorithm that sweeps the trascendental function
        for finding the eigenvalues

        Parameters
        ----------
        Bi : FLOAT
            Biot number.
        nEigen : INTEGER
            Number of Eigenvalues to lok for.
        eps : FLOAT, optional
            Small number. The default is 1e-8.

        Returns
        -------
        eigs : ARRAY, FLOAT
            Array with Eigenvalues.

        """

        # Finding out the machine epsilon
        #eps = np.finfo(float).eps

        # Initialization of container of Eigenvalues
        eigs = np.empty(nEigen) #Empty array
        eigs.fill(np.NaN)       # fill it with NaN values

        for i in range(nEigen):
            # Starting points for bisection algorithm
            if i == 0:
                xL = 0.0
            else:
                xL = (i) * np.pi - 0.5 * np.pi + eps
            xR = (i) * np.pi + 0.5 * np.pi - eps

            # find eigenvalue by bisection algorithm
            eig = bisect(xitanxi_Bi, xL, xR, args=(Bi,), full_output=True)
            eigs[i] = np.copy(eig[0])

        return eigs

    return find_eigen,


@app.cell
def __(np):
    def calcCn(eig):
        """
        Estimate the value of the Cn coefficient in the Fourier series

        Parameters
        ----------
        eig : FLOAT
            Eigenvalue.

        Returns
        -------
        FLOAT
            Cn, function.

        """
        #Cn =  (4 * np.sin(eig)) / (2*eig + np.sin(eig))
        Cn = (2 * np.sin(eig)) / (eig + np.sin(eig) * np.cos(eig))
        return Cn
    return calcCn,


@app.cell
def __(calcCn, np):
    def dimlessT(x, Fo, eigs):
        """
        Estimate a dimensionless temperature profile
        for a given Fourier number.

        Parameters
        ----------
        x : FLOAT
            Non-dimensional rod length.
        Fo : FLOAT
            Fourier number, Fo = alp*t/L2.
        eigs : FLOAT
            List with eigenvalues for summation.

        Returns
        -------
        FLOAT
            Dimensionless temperature profile.
        """
        Cn = calcCn(eigs)

        theta = 0.0

        for i, eig in enumerate(eigs):
            theta += Cn[i] * np.exp(-eig**2 * Fo) * np.cos(eig*x)

        return theta
    return dimlessT,


if __name__ == "__main__":
    app.run()
