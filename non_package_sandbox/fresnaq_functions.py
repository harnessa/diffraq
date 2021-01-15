import numpy as np

class fresnaq_functions(object):

############################################
############################################

    def polarareaquad(self,g,n,m):
        """
        % POLARAREAQUAD   quadrature for integrals over area inside polar function
        %
        % [xq yq wq] = polarareaquad(g,n,m) returns lists xq,yq of coordinates of nodes,
        %  and corresponding weights wq, for 2D quadrature over a polar domain, ie,
        %
        %    sum_{j=1}^N f(xq(j),yq(j)) wq(j)   \approx   int_Omega f(x,y) dx dy
        %
        %  for all smooth functions f, where Omega is the polar domain defined by
        %  radial function r=g(theta).
        %
        % [xq yq wq bx by] = polarareaquad(g,n,m) also returns boundary points, useful
        %  for plotting.
        %
        % Inputs:
        %   g = function handle for g(theta), theta in [0,2pi)
        %   n = # theta nodes
        %   m = # radial nodes
        %
        % Outputs:
        %  xq, yq = column vectors (real, length N) of x,y coordinates of nodes
        %  wq = column vector (real, length N) of weights
        %
        %  Note: no deriv of g func handle is needed.
        """

        #theta nodes, constant weights
        t = 2.*np.pi * np.arange(1,n+1)/n
        wt = 2.*np.pi/n
        #boundary points
        bx = np.cos(t) * g(t)
        by = np.sin(t) * g(t)
        #Rule for (0,1)
        xr, wr = self.lgwt(m, 0, 1)
        xq = np.ones((n*m,1)) * np.nan
        yq = xq.copy()
        wq = xq.copy()

        #Loop over angles
        for i in range(1,n+1):
            #This radius; index list
            r = g(t[i-1])
            jj = np.arange(m) + (i-1)*m

            #line of nodes
            xq[jj] = np.cos(t[i-1]) * r * xr
            yq[jj] = np.sin(t[i-1]) * r * xr
            #theta weight times rule for r.dr on (0,r)
            wq[jj] = wt * r**2 * xr * wr

        return xq, yq, wq, bx, by

############################################
############################################

    def lgwt(self,N,a,b):
        """
        % LGWT  Gauss-Legendre quadrature scheme on a 1D interval
        %
        % [x,w]=lgwt(N,a,b)
        %  computes the N-point Legendre-Gauss nodes x and weights w on an interval
        %  [a,b]. Both x and w are returned as column vectors, in descending order
        %  of x.
        %
        % Suppose you have a continuous function f(x) which is defined on [a,b]
        % which you can evaluate at any x in [a,b]. Simply evaluate it at all of
        % the values contained in the x vector to obtain a vector f. Then compute
        % the definite integral using sum(f.*w);

        % Written by Greg von Winckel - 02/25/2004. Docs clarify Barnett 9/4/20.
        """

        N = N-1
        N1 = N+1
        N2 = N+2

        xu = np.linspace(-1, 1, N1)[None].T

        #initial guess
        y = np.cos((2*np.arange(N+1)[None].T + 1)*np.pi/(2*N+2)) + \
            (0.27/N1)*np.sin(np.pi*xu*N/N2)

        #Legendre-Gauss Vandermonde Matrix
        L = np.zeros((N1,N2))

        #Lp = Derivative of LGVM

        #Compute the zeros of the N+1 Legendre Polynomial using the recursion relation
        # and the Newton-Raphson method

        y0=2

        #Iterate until new points are uniformly within epsilon of old points
        while max(abs(y-y0)) > np.spacing(1):

            L[:,0] = 1
            L[:,1] = y[:,0]

            for k in range(2,N1+1):
                L[:,k+1-1] = ( (2*k-1) * y.T * L[:,k-1] - (k-1)*L[:,k-1-1] ) / k

            Lp = (N2 * ( L[:,N1-1] - y.T * L[:,N2-1] ) / (1. - y.T**2))[0]

            y0=y
            y= y0 - (L[:,N2-1]/Lp)[None].T

        #Linear map from [-1,1] to [a,b]
        x = (a*(1-y) + b*(1+y))/2

        #Compute the weights
        w = ((b-a) / ((1-y.T**2)*Lp**2) * (N2/N1)**2).T

        return x, w

############################################
############################################

    def starshadequad(self,Np, Afunc, r0, r1, n, m, Aquad='g',verb=False):
        """
        % STARSHADEQUAD  quadrature for area integral over apodized petal 0-1 occulter.
        %
        % [xj yj wj bx by] = starshadequad(Np,Afunc,r0,r1,n,m,verb,Aquad)
        %
        %  Uses Theta(r) formula of (1)-(2) in Cady '12 to build area quadrature scheme
        %  over starshade, given apodization function and other geometric parameters.
        %
        % Inputs:
        %  Np = # petals
        %  Afunc = apodization from roughly 1 down to roughly 0, over domain [r0,r1].
        %  r0,r1 = apodization domain of radii (note differs from Cash'11 defn a,b)
        %   radial apodization is 1 for r<r0, A((r-r0)/(r1-r0)) for r0<r<r1, 0 for r>r1.
        %  n = nodes per petal width
        %  m = nodes over disc and over radial apodization [r0,r1]. (exact may vary)
        %  verb = (optional) 0,1, etc, verbosity.
        %  Aquad = (optional) apodization quadrature type: 'a' Alpert (default),
        %          'g' Gauss, 'u' Uniform grid (including end points).
        %
        % Outputs:
        % xj,yj - (areal) nodes
        % wj    - corresp (areal) weights
        % bx,by - boundary nodes (those lying along profile, not petal tips or gaps)
        %
        % Design is perfectly symmetric; no perturbations for now.

        % Barnett 8/24/20; changed to Afunc(r), swapped r0,r1, 9/5/20.
        """
        #Setup [0,1] quadr scheme
        if Aquad == 'g':
            #radial petal quadrature, descending z
            z, w = self.lgwt(m, 0, 1)
            z = z[:,0]
            w = w[:,0]
        elif Aquad == 'a':
            #Alpert
            z, w = self.QuadNodesInterval(0,1,m,[],1,1,32)
            #flip to descending order
            z = z[::-1]
            w = w[::-1]
            z = z[:,0]
            w = w[:,0]
        elif Aquad == 'u':
            #Uniform grid including endpoints, composite trap rule
            z = np.linspace(0,1,m)
            h = z[1]-z[0]
            z = z[::-1]
            w = np.concatenate(([0.5], np.ones(m-2), [0.5])) * h

        #Central disc
        N = int(np.ceil(0.3*n*Np))    #PTR in theta, rough guess so dx about same
        t = 2.*np.pi * np.arange(1,N+1)/N   #theta nodes, const weights
        wt = 2.*np.pi/N
        xj = np.ones(N*m)*np.nan
        yj = xj.copy()
        wj = xj.copy()
        for i in range(1,N+1):
            jj = np.arange(m) + (i-1)*m
            xj[jj] = np.cos(t[i-1]) * r0*z
            yj[jj] = np.sin(t[i-1]) * r0*z
            wj[jj] = wt*r0**2*z*w

        if verb:
            aerr = wj.sum() - np.pi*r0**2
            print(f'disc area err: {aerr:.2e}\n')

        #
        # #Petals radial outer loop (using z,w scheme from above)
        # zn, wn = self.lgwt(n, -1, 1)     #[-1,1] becomes width in theta. col vecs
        # zn = zn[:,0]
        # wn = wn[:,0]
        # r = r0 + (r1-r0)*z              #radius quadrature nodes
        # A = Afunc(r)
        # for i in range(1,m+1):
        #     wi = (r1-r0) * w[i-1] * r[i-1]
        #     if A[i-1] > 1 or A[i-1] < 0:
        #         print('0<A<1!')
        #         breakpoint()
        #
        #     t = np.kron(np.ones(Np), (np.pi/Np)*A[i-1]*zn) + \
        #         np.kron(2.*np.pi*np.arange(1,Np+1)/Np, np.ones(n)) #thetas
        #     ww = np.kron(np.ones(Np), wi*np.pi/Np*A[i-1]*wn)
        #     xj = np.concatenate((xj, r[i-1]*np.cos(t))) #Append new nodes + weights
        #     yj = np.concatenate((yj, r[i-1]*np.sin(t)))
        #     wj = np.concatenate((wj, ww))

        breakpoint()

        #Boundary points
        bx = np.ones((m,2*Np)) * np.nan
        by = bx.copy()
        r = r[::-1]
        A = A[::-1]
        for p in range(1,Np+1):
            t = 2.*np.pi/Np*p - np.pi/Np*A
            bx[:,2*p-1-1] = r*np.cos(t)
            by[:,2*p-1-1] = r*np.sin(t)
            t = 2.*np.pi/Np*p + np.pi/Np*A[::-1]
            bx[:,2*p-1] = r[::-1]*np.cos(t)
            by[:,2*p-1] = r[::-1]*np.sin(t)

        return xj, yj, wj, bx, by

############################################
############################################


############################################
############################################

############################################
############################################

############################################
############################################
