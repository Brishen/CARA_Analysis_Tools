import numpy as np

def TimeParabolaFit(t, F):
    """
    TimeParabolaFit - Calculate best-fit parabola to a function using only
                      (t,F) points nearest the minimum F point.

    Args:
        t (np.ndarray): Array of times (or x-values) associated with the parabola. [1xn]
        F (np.ndarray): Array of function values at "t" times. Must be the same size as "t". [1xn]

    Returns:
        dict: containing:
            'c': Coefficients of the best-fit parabola [3x1]
                 y = c[0]*x^2 + c[1]*x + c[2]
            'tinc': Times from the "t" array used for the best-fit parabola solution
            'Finc': Values from the "F" array used for the best-fit parabola solution
            'rankAmat': Rank of the design matrix used in creating the solution
            'x': Coefficients of the best-fit parabola using normalized times [3x1]
    """

    t = np.asarray(t).flatten()
    F = np.asarray(F).flatten()

    if t.size != F.size:
        raise ValueError('t and F must have the same size')

    # Restrict to unique times
    t_unique, indx = np.unique(t, return_index=True)
    F_unique = F[indx]

    Nt = t_unique.size
    if Nt < 3:
        raise ValueError('Minimum of three unique points required for parabolic fit')

    # Sort by ascending F value
    srt = np.argsort(F_unique)
    tvec = t_unique[srt]
    Fvec = F_unique[srt]

    # Initially include the 3 unique points with smallest F values
    inc = np.zeros(Nt, dtype=bool)
    inc[:3] = True

    # Loop to ensure full rank matrix
    while True:
        tinc = tvec[inc]

        tmin = np.min(tinc)
        tmax = np.max(tinc)
        tdel = 0.5 * (tmax - tmin)
        tmid = 0.5 * (tmax + tmin)

        if tdel == 0:
            # Should not happen with 3 unique points
            pass

        z = (tinc - tmid) / tdel

        # Design matrix [z^2 z 1]
        Amat = np.column_stack((z**2, z, np.ones_like(z)))

        # Rank check
        norm_Amat = np.linalg.norm(Amat, 2)
        if norm_Amat == 0:
             rankAtol = 0 # Should not happen
        else:
             # Matlab uses eps(X) which is distance to next floating point number from X
             # np.spacing(X) is equivalent
             rankAtol = 1000 * max(Amat.shape) * np.spacing(norm_Amat)

        rankAmat = np.linalg.matrix_rank(Amat, tol=rankAtol)

        if rankAmat >= 3 or np.all(inc):
            break

        # Add more points
        # Calculate how many points to add
        # max(1, 3 - sum(inc))
        current_count = np.sum(inc)
        num_to_add = max(1, 3 - current_count)

        # Find indices where inc is False
        candidates = np.where(~inc)[0]
        if candidates.size > 0:
            to_add = candidates[:num_to_add]
            inc[to_add] = True
        else:
            # Should be caught by np.all(inc)
            break

    # Pseudoinverse solution
    Finc = Fvec[inc]
    x_coeffs = np.linalg.pinv(Amat) @ Finc

    # Convert to coefficients of t
    trat = tmid / tdel
    x0 = x_coeffs[0]
    x1 = x_coeffs[1]
    x2 = x_coeffs[2]

    tratx0 = trat * x0

    # Calculate c coefficients
    c0 = x0 / (tdel**2)
    c1 = (x1 - 2 * tratx0) / tdel
    c2 = trat * (tratx0 - x1) + x2

    c = np.array([c0, c1, c2])

    return {
        'c': c,
        'tinc': tinc,
        'Finc': Finc,
        'rankAmat': rankAmat,
        'x': x_coeffs
    }
