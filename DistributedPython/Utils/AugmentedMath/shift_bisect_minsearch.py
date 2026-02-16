import numpy as np

def shift_bisect_minsearch(fun, x0, y0, Nbisectmax=100, Nshiftmax=100, tolX=1e-4, tolY=1e-4, verbose=1):
    """
    Find the minimum of the function "y = fun(x)" given the initial search interval (x1,x2).

    Because this function uses an algorithm that performs both shifting and
    bisection in search for the minimum, (x1,x2) does not necessarily have to
    bound the minimum.

    This function finds one and only one minimum, so if there are multiple
    minima over (x1,x2), then this function will only find one at most.

    Args:
        fun (callable): Function to minimize.
        x0 (array-like): Initial grid points (must be 5 or more).
        y0 (array-like): Initial function values at x0.
        Nbisectmax (int, optional): Max number of bisections. Defaults to 100.
        Nshiftmax (int, optional): Max number of shifting iterations. Defaults to 100.
        tolX (float, optional): Tolerance for x convergence. Defaults to 1e-4.
        tolY (float, optional): Tolerance for y convergence. Defaults to 1e-4.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        tuple: (xmin, ymin, converged, nbisect, nshift, xbuf, ybuf)
    """

    # Initial number of grid points (should be 5 or larger)
    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    N0 = x0.size
    if N0 < 5:
        raise ValueError('shift_bisect_minsearch requires five or more points')

    if N0 != y0.size:
        raise ValueError('shift_bisect_minsearch needs same number of (x,y) grid points')

    if np.any(np.diff(x0) < 0):
        raise ValueError('shift_bisect_minsearch needs monotonically increasing grid points')

    # Defaults for optional inputs handled by python arguments

    if tolY is not None and (np.isnan(tolY) or tolY < 0):
        raise ValueError('Invalid tolY parameter')

    buffering = True # Always buffer in Python implementation for simplicity

    # Find minimum value over initial grid
    if np.max(y0) != np.min(y0):
        # Find min point
        nmin = np.argmin(y0)
    else:
        # Use center point if all initial y values are equal
        nmin = int(round((N0-1)/2))
        # Note: MATLAB is 1-based, Python is 0-based.
        # MATLAB: round((Ninitial-1)/2+1) -> index.
        # Python: round((N0-1)/2) -> index.

    # Initialize the 5-point bisection search buffer
    # Python indices are 0-based.

    if nmin < 2:
        # Minimum is near or at left boundary
        n1 = 0
        n2 = 5 # Exclusive upper bound for slice
    elif nmin > N0 - 3:
        # Minimum is near or at right boundary
        n1 = N0 - 5
        n2 = N0 # Exclusive upper bound
    else:
        # Minimum point in center somewhere
        n1 = nmin - 2
        n2 = nmin + 3 # Exclusive upper bound

    x = x0[n1:n2].copy()      # Holds 5 x values
    y = y0[n1:n2].copy()      # Holds 5 y values
    c = np.zeros(x.shape, dtype=bool)  # Holds 5 logical values (true => y needs to be calculated)

    # Initialize the buffers to hold new variables
    xnew = np.zeros(x.shape)
    ynew = np.zeros(xnew.shape)
    cnew = np.ones(x.shape, dtype=bool)

    # Perform iterative search
    if verbose > 0:
        bufstr = ' (with output buffering)' if buffering else ''
        print(f'Performing shifting + bisection search for minimum{bufstr}')

    # Iteration and convergence flags
    still_iterating  = True
    converged = False

    nshift = 0  # Counts shifts to left or right
    nbisect = 0 # Counts bisections

    xbuf = x0.copy()
    ybuf = y0.copy()

    # Iterate until converged or stopped
    while still_iterating:

        # Calculate any required y values
        for n in range(5):
            if c[n]:
                y[n] = fun(x[n])
                c[n] = False
                if buffering:
                    xbuf = np.append(xbuf, x[n])
                    ybuf = np.append(ybuf, y[n])

        # Find min among the 5 calculated values
        nmin = np.argmin(y)
        ymin = y[nmin]

        if ymin == np.max(y):
            # Use center point if all y values are equal
            nmin = 2 # Index 2 is center of 0..4

        xmin = x[nmin]

        if verbose > 1:
            print(f'  (x,y) = {x[nmin]} {y[nmin]}')

        # Either shift if min is on edge, or bisect if min is interior

        if nmin == 0:
            # First x point is the lowest y value (perform shift to left)
            if verbose > 1:
                print(f' Shifting left, nshift = {nshift}')

            xnew[1:5] = x[0:4]
            ynew[1:5] = y[0:4]
            cnew[1:5] = False

            dx        = x[1] - x[0]
            xnew[0]   = x[0] - dx
            cnew[0]   = True

            nshift    = nshift + 1

        elif nmin == 4:
            # Last x point is the highest y value (perform shift to right)
            if verbose > 1:
                print(f' Shifting right, nshift = {nshift}')

            xnew[0:4] = x[1:5]
            ynew[0:4] = y[1:5]
            cnew[0:4] = False

            dx        = x[4] - x[3]
            xnew[4]   = x[4] + dx
            cnew[4]   = True

            nshift    = nshift + 1

        else:
            # One of the interior points has the highest y value (lowest actually, looking for min),
            # so check for convergence, and set up to perform next bisection if required

            nlo = nmin - 1
            nhi = nmin + 1

            # Check for convergence

            dxlo = abs(x[nmin] - x[nlo])
            dxhi = abs(x[nmin] - x[nhi])
            dx   = max(dxlo, dxhi)

            if verbose > 2:
                tolX_str = str(dx/tolX) if tolX is not None and tolX != 0 else 'Inf'
                print(f' dx = {dx} dx/tolX = {tolX_str}')

            # Use machine epsilon for x[nmin] if it's float
            epsX = np.finfo(float).eps * abs(x[nmin]) if x[nmin] != 0 else np.finfo(float).eps

            if (tolX is not None and dx < tolX) or (dx < 10 * epsX):
                converged = True
                if verbose > 0:
                    print(f'Convergence achieved because dx < tolX or dx < epsX (nbisect = {nbisect} nshift = {nshift})')
            else:
                dylo = y[nlo] - y[nmin]
                dyhi = y[nhi] - y[nmin]
                dy   = max(dylo, dyhi)

                if verbose > 2 and tolY is not None and not np.isnan(tolY):
                    print(f' dy = {dy} dy/tolY = {dy/tolY}')

                if tolY is not None and not np.isnan(tolY) and dy < tolY:
                    converged = True
                    if verbose > 0:
                        print(f'Convergence achieved because dy < tolY (nbisect = {nbisect} nshift = {nshift})')

            # If not yet converged then set up to perform next bisection
            if not converged:
                xnew[0] = x[nlo]
                ynew[0] = y[nlo]
                cnew[0] = False

                xnew[1] = (x[nlo] + x[nmin]) / 2.0
                cnew[1] = True

                xnew[2] = x[nmin]
                ynew[2] = y[nmin]
                cnew[2] = False

                xnew[3] = (x[nhi] + x[nmin]) / 2.0
                cnew[3] = True

                xnew[4] = x[nhi]
                ynew[4] = y[nhi]
                cnew[4] = False

                nbisect = nbisect + 1

        # Check for NaN values, or too many shift or bisection iterations
        if np.any(np.isnan(y)):
            # NaN values detected
            if verbose > 0:
                print('NaN y-values detected; returning unconverged result')
            still_iterating = False

        elif nbisect > Nbisectmax:
            # Too many bisections
            if verbose > 0:
                print('Maximum number of bisections exceeded; returning unconverged result')
            still_iterating = False

        elif nshift > Nshiftmax:
            # Too many shifts
            if verbose > 0:
                print('Maximum number of shifts exceeded; returning unconverged result')
            still_iterating = False

        # Check for convergence
        if converged:
            # Stop iterating if converged
            still_iterating = False
        else:
            # Copy the new buffers to set up for next iteration
            x = xnew.copy()
            y = ynew.copy()
            c = cnew.copy()

    # Sort the buffered values
    if buffering:
        srt = np.argsort(xbuf)
        xbuf = xbuf[srt]
        ybuf = ybuf[srt]

    return xmin, ymin, converged, nbisect, nshift, xbuf, ybuf
