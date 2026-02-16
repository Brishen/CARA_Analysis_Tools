import numpy as np

def eig2x2(Araw):
    """
    eig2x2 - eigenvalue and eigenvector solver for 2x2 symmetric matrices.

    Args:
        Araw (np.ndarray): matrix representing n symmetric matrices [nx3]
                           Araw[:,0] is the 1,1 component
                           Araw[:,1] is the 1,2 and 2,1 component
                           Araw[:,2] is the 2,2 component

    Returns:
        tuple:
            V1 (np.ndarray): matrix representing n of the 1st eigenvectors [nx2]
            V2 (np.ndarray): matrix representing n of the 2nd eigenvectors [nx2]
            L1 (np.ndarray): matrix representing n of the 1st eigenvalues [nx1]
            L2 (np.ndarray): matrix representing n of the 2nd eigenvalues [nx1]
    """
    Araw = np.asarray(Araw)

    # Handle 1D input (single matrix)
    if Araw.ndim == 1:
        Araw = Araw.reshape(1, -1)

    Nvec = Araw.shape[0]

    # Calculate eigenvalues and eigenvectors for each covariance matrix
    #   A = [a b; c d], with b = c
    # with
    #   a = Amat(:,0); b = Amat(:,1); c = b; d = Amat(:,2);
    #   Trace: T = a+d
    #   Determinant: D = a*d-b*c

    T = Araw[:, 0] + Araw[:, 2]
    D = Araw[:, 0] * Araw[:, 2] - Araw[:, 1]**2

    # Eigenvalues calculation
    # Use max(0, ...) inside sqrt to avoid numerical negative values close to 0
    # resulting in NaNs, though here T^2 - 4D should be >= 0 for symmetric real matrices.
    # Discriminant delta = T^2 - 4D = (a+d)^2 - 4(ad-b^2) = a^2 + 2ad + d^2 - 4ad + 4b^2 = (a-d)^2 + 4b^2 >= 0.
    discriminant = T**2 - 4*D
    # Ensure non-negative due to floating point errors
    discriminant[discriminant < 0] = 0

    sqrt_delta = np.sqrt(discriminant)

    L1 = (T + sqrt_delta) / 2.0 # Largest eigenvalue
    L2 = (T - sqrt_delta) / 2.0 # Smallest eigenvalue

    # Initialize the conjunction plane eigenvector arrays
    V1 = np.full((Nvec, 2), np.nan)
    V2 = np.full((Nvec, 2), np.nan)

    # Eigenvectors for the subset of covariance matrices that have
    # non-zero off-diagonal values
    c0 = Araw[:, 1] != 0

    # Indices where c0 is true
    if np.any(c0):
        # We can use boolean indexing
        # V1[c0, 0] = L1[c0] - Araw[c0, 2]
        # V2[c0, 0] = L2[c0] - Araw[c0, 2]
        # V1[c0, 1] = Araw[c0, 1]
        # V2[c0, 1] = Araw[c0, 1]

        # NOTE: MATLAB code:
        # V1(c0,1) = L1(c0)-Araw(c0,3); -> Python index 2
        # V2(c0,1) = L2(c0)-Araw(c0,3); -> Python index 2
        # V1(c0,2) = Araw(c0,2); -> Python index 1
        # V2(c0,2) = Araw(c0,2); -> Python index 1

        V1[c0, 0] = L1[c0] - Araw[c0, 2]
        V2[c0, 0] = L2[c0] - Araw[c0, 2]
        V1[c0, 1] = Araw[c0, 1]
        V2[c0, 1] = Araw[c0, 1]

        # Normalization
        norm1 = np.sqrt(V1[c0, 0]**2 + V1[c0, 1]**2)
        norm2 = np.sqrt(V2[c0, 0]**2 + V2[c0, 1]**2)

        # Avoid division by zero if norm is zero (should not happen if c0 is true and distinct eigenvalues?)
        # If eigenvalues are equal and b!=0, then a=d?
        # (a-d)^2 + 4b^2 = 0 => a=d and b=0. But c0 says b!=0. So distinct eigenvalues.

        # Reshape norms for broadcasting
        # repmat in MATLAB: [1 2] -> repeats columns.
        # Python broadcasting: (N, 1) / (N, 1)

        V1[c0, :] = V1[c0, :] / norm1[:, np.newaxis]
        V2[c0, :] = V2[c0, :] / norm2[:, np.newaxis]

    # Eigenvectors for A matrices with zero off-diagonal values
    c0_not = ~c0
    if np.any(c0_not):
        # ca = Araw(:,3) <= Araw(:,1); -> d <= a
        ca = Araw[:, 2] <= Araw[:, 0]

        c1 = c0_not & ca
        # V1(c1,1) = 1; V2(c1,1) = 0;
        # V1(c1,2) = 0; V2(c1,2) = 1;
        if np.any(c1):
            V1[c1, 0] = 1.0; V2[c1, 0] = 0.0
            V1[c1, 1] = 0.0; V2[c1, 1] = 1.0

        c2 = c0_not & ~ca
        # V1(c1,1) = 0; V2(c1,1) = 1;
        # V1(c1,2) = 1; V2(c1,2) = 0;
        if np.any(c2):
            V1[c2, 0] = 0.0; V2[c2, 0] = 1.0
            V1[c2, 1] = 1.0; V2[c2, 1] = 0.0

    # Find instances where the determinant is not accurately calculated
    # due to floating point representation errors
    # floatErrIdx = (Araw(:,1).*Araw(:,3) == Araw(:,1).*Araw(:,3)-Araw(:,2).^2) & Araw(:,2).^2 ~= 0;

    term1 = Araw[:, 0] * Araw[:, 2]
    term2 = Araw[:, 1]**2
    floatErrIdx = (term1 == (term1 - term2)) & (term2 != 0)

    # Check for "b" values that are close to 0, but aren't exactly 0.
    bVal0Idx = (np.abs(Araw[:, 1]) < 1e-2) & (Araw[:, 1] != 0)

    # Manually run eig() for any of the special checks
    c_fix = floatErrIdx | bVal0Idx

    if np.any(c_fix):
        fix_indices = np.where(c_fix)[0]
        for idx in fix_indices:
            # tempA = [Araw(idx,1) Araw(idx,2); Araw(idx,2) Araw(idx,3)];
            tempA = np.array([[Araw[idx, 0], Araw[idx, 1]],
                              [Araw[idx, 1], Araw[idx, 2]]])

            # [V, D] = eig(tempA, 'vector');
            # [D, ind] = sort(D);
            # V = V(:, ind);

            D, V = np.linalg.eigh(tempA)
            # eigh returns eigenvalues in ascending order, so D[0] is smallest, D[1] is largest

            # MATLAB:
            # L2(idx) = D(1); -> Smallest
            # L1(idx) = D(2); -> Largest
            # V2(idx,:) = V(:,1)'; -> Smallest eigenvector
            # V1(idx,:) = V(:,2)'; -> Largest eigenvector

            L2[idx] = D[0]
            L1[idx] = D[1]
            V2[idx, :] = V[:, 0]
            V1[idx, :] = V[:, 1]

    return V1, V2, L1, L2
