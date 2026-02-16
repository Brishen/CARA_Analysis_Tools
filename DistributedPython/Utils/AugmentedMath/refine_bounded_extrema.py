import numpy as np
import warnings
from .extrema import extrema

def refine_bounded_extrema(fun, x1, x2, Ninitial=None, Nbisectmax=100, extrema_types=3,
                           TolX=1e-6, TolY=1e-6, endpoints=False, verbose=0, check_inputs=True):
    """
    Find and refine all of the the extrema of the function "y = fun(x)"
    within the interval (x1,x2) and initially divided up into Ninitial points.

    Args:
        fun (callable or None): Anonymous function used to find and refine extrema.
                                Can be None if Mode 2 is used (implicit in MATLAB).
        x1 (float or np.ndarray):
            MODE 1: Scalar holding the x start bound.
            MODE 2: Vector holding the x values.
        x2 (float or np.ndarray):
            MODE 1: Scalar holding the x end bound.
            MODE 2: Vector holding the y values corresponding to x1.
        Ninitial (int or None):
            MODE 1: Number of initial points spanning (x1,x2) inclusively.
            MODE 2: None (must be None or empty).
        Nbisectmax (int, optional): Maximum number of bisections allowed. Defaults to 100.
        extrema_types (int, optional): 1 for only minima, 2 for only maxima, and 3 for both. Defaults to 3.
        TolX (float or list/tuple, optional): Tolerance for x-value convergence. Defaults to 1e-6.
        TolY (float or list/tuple, optional): Tolerance for y-value convergence. Defaults to 1e-6.
        endpoints (bool, optional): Flag indicating that extrema at endpoints should also be refined. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.
        check_inputs (bool, optional): Flag to suppress default and error checking. Defaults to True.

    Returns:
        tuple: (xmnma, ymnma, xmxma, ymxma, converged, nbisect, x, y, imnma, imxma)
    """

    xmnma = np.array([])
    ymnma = np.array([])
    imnma = np.array([], dtype=int)
    xmxma = np.array([])
    ymxma = np.array([])
    imxma = np.array([], dtype=int)

    nbisect = 0
    converged = False

    # Check run mode
    mode_1 = Ninitial is not None

    if check_inputs:
        if extrema_types < 1 or extrema_types > 3:
            raise ValueError('Illegal value for extrema_types: 1 for only minima, 2 for only maxima, and 3 for both.')

        if mode_1:
            # Mode where (x1,x2) specify the bounding endpoints
            if not np.isscalar(Ninitial) or Ninitial < 5:
                warnings.warn('MODE 1: Ninitial too small (must be 5 or larger); setting Ninitial = 5.')
                Ninitial = 5

            if not np.isscalar(x1) or np.isinf(x1) or np.isnan(x1):
                raise ValueError('MODE 1: Illegal value for x1 bound.')

            if not np.isscalar(x2) or np.isinf(x2) or np.isnan(x2):
                raise ValueError('MODE 1: Illegal value for x2 bound.')
        else:
            # Mode where (x1,x2) specify the initial (x,y) vectors
            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            Nx = x1.size
            Ny = x2.size

            if Nx != Ny:
                raise ValueError('MODE 2: Input (x,y) vectors have unequal dimensions')
            elif Nx < 5:
                warnings.warn('MODE 2: Input (x,y) vectors have fewer than 5 points; recommend at least 5 for stable operation.')

    # Tolerances handling
    if np.isscalar(TolX):
        TolX = [TolX, np.nan]
    else:
        TolX = list(TolX)
        if len(TolX) == 1:
            TolX.append(np.nan)

    if np.isscalar(TolY):
        TolY = [TolY, np.nan]
    else:
        TolY = list(TolY)
        if len(TolY) == 1:
            TolY.append(np.nan)

    AbsTolXConsidered = not np.isnan(TolX[0])
    RelTolXConsidered = not np.isnan(TolX[1])
    AbsTolYConsidered = not np.isnan(TolY[0])
    RelTolYConsidered = not np.isnan(TolY[1])

    NoTolXConsidered = not AbsTolXConsidered and not RelTolXConsidered
    NoTolYConsidered = not AbsTolYConsidered and not RelTolYConsidered

    if NoTolXConsidered and NoTolYConsidered:
        raise ValueError('Tolerance in X and/or Y must be specified')

    AllTolXConsidered = AbsTolXConsidered and RelTolXConsidered
    AllTolYConsidered = AbsTolYConsidered and RelTolYConsidered

    # Set up initial (x,y) grid
    if mode_1:
        if verbose > 0:
            print(f" Calculating {Ninitial} initial points.")

        minx = min(x1, x2)
        maxx = max(x1, x2)

        if minx == maxx:
            raise ValueError('MODE 1: (x1,x2) bounds cannot be equal.')

        x = np.linspace(minx, maxx, int(Ninitial))
        y = fun(x)
    else:
        x = x1
        y = x2
        if verbose > 0:
            print(f" Using {x.size} initial input points.")

        if check_inputs:
            diffx = np.diff(x)
            if np.any(diffx == 0):
                raise ValueError('MODE 2: Input x vector cannot have redundant values.')
            elif np.min(diffx) <= 0:
                warnings.warn('MODE 2: Input x vector not sorted - sorting.')
                nsrt = np.argsort(x)
                x = x[nsrt]
                y = y[nsrt]

    still_refining = True

    while still_refining:
        # Find all interior extrema
        if extrema_types == 1: # only minima
            _, _, xmnma, imnma = extrema(y, include_endpoints=endpoints)
            xmnma = x[imnma]
            ymnma = y[imnma]
            xmxma = np.array([])
            ymxma = np.array([])
            imxma = np.array([], dtype=int)
        elif extrema_types == 2: # only maxima
            xmxma, imxma, _, _ = extrema(y, include_endpoints=endpoints)
            xmxma = x[imxma]
            ymxma = y[imxma]
            xmnma = np.array([])
            ymnma = np.array([])
            imnma = np.array([], dtype=int)
        elif extrema_types == 3: # both
            # Note: extrema returns indices into y.
            # extrema signature: returns (xmax, imax, xmin, imin)
            # where output xmax/xmin are values from input vector (y in this case)
            # and imax/imin are indices into y.
            ymxma, imxma, ymnma, imnma = extrema(y, include_endpoints=endpoints)
            xmnma = x[imnma]
            xmxma = x[imxma]

        # Combined extrema indices
        iexma = np.concatenate((imnma, imxma))
        Nexma = iexma.size

        if Nexma == 0:
            still_refining = False
            converged = True
            if verbose:
                print('WARNING: No interior extrema found')
        else:
            N = x.size
            Nm1 = N - 1

            nnew = 0
            xnew = []
            DelXeps = False

            for nnmd in range(Nexma):
                # Mid-point for bisection
                # Ensure we don't go out of bounds (though extrema should handle this)
                # In Python, indices are 0 to N-1.
                # MATLAB: nmd = max(2,min(Nm1,iexma(nnmd))); -> 2 to N-1 (1-based)
                # Python: max(1, min(N-2, idx)) (0-based)
                # But iexma contains indices from extrema, which can include endpoints (0 and N-1).
                # If endpoint is 0, we can't look at left neighbor (-1).
                # If endpoint is N-1, we can't look at right neighbor (N).

                idx = iexma[nnmd]
                nmd = max(1, min(N - 2, idx))

                # Low and high x-value points
                nlo = nmd - 1
                DelXlo = x[nmd] - x[nlo]
                nhi = nmd + 1
                DelXhi = x[nhi] - x[nmd]

                # Bisect low-X segment if required
                DelX = DelXlo
                if DelX <= max(np.finfo(float).eps * np.abs(x[nmd]), np.finfo(float).eps * np.abs(x[nlo])): # approximate eps check
                    DelXeps = True

                if NoTolXConsidered:
                    if DelXeps:
                        DelXabovetol = False
                    else:
                        DelXabovetol = DelXlo > 2 * DelXhi
                else:
                    ref_val = max(abs(x[nmd]), abs(x[nlo]))
                    if AllTolXConsidered:
                        DelXabovetol = (DelX > TolX[0]) and (DelX > TolX[1] * ref_val)
                    elif AbsTolXConsidered:
                        DelXabovetol = (DelX > TolX[0])
                    else:
                        DelXabovetol = (DelX > TolX[1] * ref_val)

                if NoTolYConsidered:
                    DelYabovetol = False
                else:
                    DelY = abs(y[nmd] - y[nlo])
                    ref_val_y = max(abs(y[nmd]), abs(y[nlo]))
                    if AllTolYConsidered:
                        DelYabovetol = (DelY > TolY[0]) and (DelY > TolY[1] * ref_val_y)
                    elif AbsTolYConsidered:
                        DelYabovetol = (DelY > TolY[0])
                    else:
                        DelYabovetol = (DelY > TolY[1] * ref_val_y)

                if DelXabovetol or DelYabovetol:
                    nnew += 1
                    xnew.append((x[nmd] + x[nlo]) / 2)

                # Bisect high-X segment if required
                DelX = DelXhi
                if DelX <= max(np.finfo(float).eps * np.abs(x[nmd]), np.finfo(float).eps * np.abs(x[nhi])):
                    DelXeps = True

                if NoTolXConsidered:
                    if DelXeps:
                        DelXabovetol = False
                    else:
                        DelXabovetol = DelXhi > 2 * DelXlo
                else:
                    ref_val = max(abs(x[nhi]), abs(x[nmd]))
                    if AllTolXConsidered:
                        DelXabovetol = (DelX > TolX[0]) and (DelX > TolX[1] * ref_val)
                    elif AbsTolXConsidered:
                        DelXabovetol = (DelX > TolX[0])
                    else:
                        DelXabovetol = (DelX > TolX[1] * ref_val)

                if NoTolYConsidered:
                    DelYabovetol = False
                else:
                    DelY = abs(y[nhi] - y[nmd])
                    ref_val_y = max(abs(y[nhi]), abs(y[nmd]))
                    if AllTolYConsidered:
                        DelYabovetol = (DelY > TolY[0]) and (DelY > TolY[1] * ref_val_y)
                    elif AbsTolYConsidered:
                        DelYabovetol = (DelY > TolY[0])
                    else:
                        DelYabovetol = (DelY > TolY[1] * ref_val_y)

                if DelXabovetol or DelYabovetol:
                    nnew += 1
                    xnew.append((x[nmd] + x[nhi]) / 2)

            # Check for convergence
            if nnew == 0:
                converged = True
                still_refining = False
            elif DelXeps:
                converged = False
                still_refining = False
                if verbose > 0:
                    print('WARNING: Bisected delta-x comparable to epsilon, possibly due to noisy/discontinuous objective function')

            if still_refining:
                if verbose > 0:
                    print(f'Bisection iteration {nbisect+1} requires {nnew} new points')

                xnew = np.unique(xnew)
                # Only call fun if we have new points and fun is defined (Mode 1)
                # In Mode 2, fun is None? No, refine_bounded_extrema doc says fun used to refine.
                # In Mode 2, x1, x2 are input points. But refining requires evaluating fun at new points.
                # The MATLAB code:
                # if mode_1: y = fun(x)
                # else: x=x1, y=x2
                # But inside the loop: ynew = fun(xnew).
                # So even in Mode 2, `fun` must be provided!

                if fun is None:
                    raise ValueError("Function 'fun' must be provided for refinement even in Mode 2.")

                ynew = fun(xnew)

                x = np.concatenate((x, xnew))
                y = np.concatenate((y, ynew))

                nsrt = np.argsort(x)
                x = x[nsrt]
                y = y[nsrt]

                nbisect += 1
                if nbisect > Nbisectmax:
                    still_refining = False
                    if verbose > 0:
                        print('WARNING: Maximum number of bisections exceeded')

                    # Recalculate extrema
                    if extrema_types == 1:
                        _, _, xmnma, imnma = extrema(y, include_endpoints=endpoints)
                        xmnma = x[imnma]
                        ymnma = y[imnma]
                    elif extrema_types == 2:
                        xmxma, imxma, _, _ = extrema(y, include_endpoints=endpoints)
                        xmxma = x[imxma]
                        ymxma = y[imxma]
                    elif extrema_types == 3:
                        ymxma, imxma, ymnma, imnma = extrema(y, include_endpoints=endpoints)
                        xmnma = x[imnma]
                        xmxma = x[imxma]

    if verbose > 0:
        if converged:
            print(f'Convergence achieved after {nbisect} iterations')
        else:
            print(f'WARNING: No convergence achieved after {nbisect} iterations')

    return xmnma, ymnma, xmxma, ymxma, converged, nbisect, x, y, imnma, imxma
