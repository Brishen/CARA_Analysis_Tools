import numpy as np

def extrema(x, include_endpoints=True, sort_output=False):
    """
    Find extrema from a series of points, optionally including the endpoints
    in the search.

    Args:
        x (array-like): Input vector of points.
        include_endpoints (bool, optional): Whether to include endpoints in the search. Defaults to True.
        sort_output (bool, optional): Whether to sort the output by value (descending for max, ascending for min). Defaults to False.

    Returns:
        tuple: (xmax, imax, xmin, imin)
            xmax (np.ndarray): Maxima points.
            imax (np.ndarray): Indices of the maxima.
            xmin (np.ndarray): Minima points.
            imin (np.ndarray): Indices of the minima.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Entry must be a vector.")

    Nt = x.size
    indx = np.arange(Nt)

    # Handle NaNs
    inan = np.isnan(x)
    if np.any(inan):
        indx = indx[~inan]
        x = x[~inan]
        Nt = x.size

    if Nt == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Difference between subsequent elements
    dx = np.diff(x)

    # Horizontal line
    if not np.any(dx):
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Use the middle element for flat peaks
    # Indexes where x changes (non-zero diff)
    a = np.flatnonzero(dx != 0)

    # Indexes where a does not change (consecutive non-zero diffs)
    # diff(a) != 1 means there was a gap (a zero diff) between indices in `a`.
    lm = np.flatnonzero(np.diff(a) != 1) + 1
    d = a[lm] - a[lm-1] # Number of elements in the flat peak (gap size)
    a[lm] = a[lm] - np.floor(d / 2).astype(int) # Save middle elements

    # Add end point (Nt-1 in 0-based indexing corresponds to Nt in MATLAB 1-based)
    # a holds indices into dx, so max index is Nt-2.
    # But for finding peaks, we essentially map back to x indices.
    a = np.append(a, Nt - 1)

    # Peaks
    xa = x[a]           # Series without flat peaks
    b = (np.diff(xa) > 0).astype(int) # 1 => positive slopes (minima begin), 0 => negative slopes (maxima begin)
    xb = np.diff(b)     # -1 => maxima indexes (but one), +1 => minima indexes (but one)

    imax_idx = np.flatnonzero(xb == -1) + 1 # maxima indexes in `a`
    imin_idx = np.flatnonzero(xb == +1) + 1 # minima indexes in `a`

    imax = a[imax_idx]
    imin = a[imin_idx]

    nmaxi = imax.size
    nmini = imin.size

    # Add endpoints if required
    if include_endpoints:
        # Maximum or minimum on a flat peak at the ends?
        if nmaxi == 0 and nmini == 0:
            if x[0] > x[-1]:
                xmax = np.array([x[0]])
                imax = np.array([0]) # Index into cleaned x
                xmin = np.array([x[-1]])
                imin = np.array([Nt - 1]) # Index into cleaned x
            elif x[0] < x[-1]:
                xmax = np.array([x[-1]])
                imax = np.array([Nt - 1])
                xmin = np.array([x[0]])
                imin = np.array([0])
            else:
                xmax = np.array([])
                imax = np.array([], dtype=int)
                xmin = np.array([])
                imin = np.array([], dtype=int)

            # Map back
            final_imax = indx[imax]
            final_imin = indx[imin]
            return xmax, final_imax, xmin, final_imin

        # Maximum or minimum at the ends?
        if nmaxi == 0:
            imax = np.array([0, Nt - 1])
        elif nmini == 0:
            imin = np.array([0, Nt - 1])
        else:
            if imax[0] < imin[0]:
                imin = np.insert(imin, 0, 0)
            else:
                imax = np.insert(imax, 0, 0)

            if imax[-1] > imin[-1]:
                imin = np.append(imin, Nt - 1)
            else:
                imax = np.append(imax, Nt - 1)

    # Map back to original indices if NaNs were removed
    final_imax = indx[imax]
    final_imin = indx[imin]

    xmax = x[imax]
    xmin = x[imin]

    if sort_output:
        # Descending order for max
        inmax = np.argsort(-xmax)
        xmax = xmax[inmax]
        final_imax = final_imax[inmax]

        # Ascending order for min
        inmin = np.argsort(xmin)
        xmin = xmin[inmin]
        final_imin = final_imin[inmin]

    return xmax, final_imax, xmin, final_imin
