import numpy as np
import warnings
from .CovRemEigValClip2x2 import CovRemEigValClip2x2

def RemediateCovariance2x2(ProjectedCov, HBR, WarningLevel=0):
    """
    RemediateCovariance2x2 - Attempts to remediate any non-positive definite
                             2x2 covariances into positive definite
                             covariances. The method is optimized for
                             vectorized operations.

    Args:
        ProjectedCov (np.ndarray): Represents a series of n 2x2 symmetric covariances
                                   which may need to be remediated.
                                   (nx3 row vectors for n covariances represented as
                                    [(1,1) (2,1) (2,2])
        HBR (np.ndarray or float): Hard body region (nx1 matrix of HBR values or scalar)
        WarningLevel (int, optional): Specifies warnings issued. Defaults to 0.

    Returns:
        tuple:
            Arem (np.ndarray): Remediated covariance. (nx3) row vectors. NaN if failed.
            RevCholCov (np.ndarray): Reverse Cholesky factorization. (nx3). NaN if failed.
            IsPosDef (np.ndarray): Flag indicating if PD (nx1 bool).
            IsRemediated (np.ndarray): Flag indicating if remediated (nx1 bool).
    """

    ProjectedCov = np.asarray(ProjectedCov, dtype=float)

    # Handle single matrix input (1D array)
    if ProjectedCov.ndim == 1:
        ProjectedCov = ProjectedCov.reshape(1, -1)

    numCov = ProjectedCov.shape[0]

    # Ensure HBR is compatible
    HBR = np.asarray(HBR, dtype=float)
    if HBR.ndim == 0:
        HBR = np.full(numCov, HBR)
    else:
        HBR = HBR.reshape(-1)
        if HBR.size != numCov:
             # If HBR is a single value array, broadcast it
             if HBR.size == 1:
                 HBR = np.full(numCov, HBR[0])
             else:
                 raise ValueError(f"HBR size {HBR.size} must match number of covariances {numCov}")

    # Fill output Arem array
    Arem = ProjectedCov.copy()

    # Preallocate outputs
    IsPosDef = np.ones(numCov, dtype=bool)
    IsRemediated = np.zeros(numCov, dtype=bool)

    # Initial Check
    NPD_PosDef2x2 = _PosDef2x2(ProjectedCov) != 0
    RevCholCov = _RevChol2x2(ProjectedCov)

    # Check for imaginary components in RevCholCov
    # np.imag returns imaginary part.
    NPD_RevChol2x2 = np.max(np.abs(np.imag(RevCholCov)), axis=1) != 0

    NPD = NPD_PosDef2x2 | NPD_RevChol2x2

    # If there are any NPD covariances, attempt to fix them
    numNPD = np.sum(NPD)

    if numNPD > 0:
        # Allowable remediation eigenvalue clipping factors
        Fclip = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
        Nclip = Fclip.size
        numFixed = np.zeros(Nclip, dtype=int)

        idxNPD = NPD.copy() # Boolean mask of currently NPD matrices

        # We need to track the latest RevCholCov for the iterative process
        # But RevCholCov is updated at the end of loop for fixed ones.

        for nclip in range(Nclip):
            # Indices where we need to attempt remediation
            # idxNPD is a boolean mask of length numCov

            # Remediate NPD covariance using current clipping factor
            # Lclip = (Fclip * HBR)^2
            current_HBR = HBR[idxNPD]
            Lclip_vals = (Fclip[nclip] * current_HBR)**2

            # Call CovRemEigValClip2x2
            # It returns ClipStatus, Arem_subset
            # Note: CovRemEigValClip2x2 returns ClipStatus as boolean, and Remediated Matrices
            # But here IsRemediated tracks if it was EVER remediated?
            # MATLAB: [IsRemediated(idxNPD),Arem(idxNPD,:)] = CovRemEigValClip2x2(...)
            # So IsRemediated is updated for the subset.

            subset_ClipStatus, subset_Arem = CovRemEigValClip2x2(ProjectedCov[idxNPD], Lclip_vals)

            # Update IsRemediated for the subset
            IsRemediated[idxNPD] = subset_ClipStatus
            Arem[idxNPD] = subset_Arem

            # Check NPD status of remediated covariance
            subset_Arem_check = Arem[idxNPD]

            subset_NPD_PosDef2x2 = _PosDef2x2(subset_Arem_check) != 0
            subset_ReChCov = _RevChol2x2(subset_Arem_check)
            subset_NPD_RevChol2x2 = np.max(np.abs(np.imag(subset_ReChCov)), axis=1) != 0

            subset_NPD = subset_NPD_PosDef2x2 | subset_NPD_RevChol2x2

            # Update main NPD mask
            # idxNPD is the mask of those processing in this step.
            # subset_NPD is boolean array relative to that subset.
            # We want to identify which ones are NOW fixed.
            # Fixed means they were in idxNPD (so True) and now subset_NPD is False.

            # Let's map subset_NPD back to full size
            # new_NPD for the whole array
            # Start with current NPD status (which is idxNPD)
            # If subset_NPD is False, then NPD becomes False.

            # Construct a full size False array and fill it?
            # No, easiest is to use indices.

            # Get indices of True in idxNPD
            indices_processing = np.nonzero(idxNPD)[0]

            # Update IsPosDef
            # subset_NPD is True if NPD (failed).
            # IsPosDef should be True if not NPD.
            IsPosDef[indices_processing] = ~subset_NPD

            # Identify newly fixed
            # Fixed in this iteration = (was NPD) AND (now PD)
            # In MATLAB: updateIdxs = NPD == 0 & idxNPD == 1; (where idxNPD was the mask at start of loop)
            # Here idxNPD is mask at start.
            # subset_NPD corresponds to indices_processing.

            # Map subset_NPD back to a full boolean mask 'current_step_NPD_status' (for the ones processed)
            # or just update the main NPD mask.

            # We want to count how many fixed *in this iteration*.
            # fixed_in_this_step = ~subset_NPD
            num_fixed_this_step = np.sum(~subset_NPD)
            numFixed[nclip] = num_fixed_this_step

            # Update ProjectedCov for fixed ones
            # MATLAB: ProjectedCov(updateIdxs,:) = Arem(updateIdxs,:);
            # updateIdxs are those in indices_processing where subset_NPD is False.

            fixed_indices_subset = np.where(~subset_NPD)[0]
            fixed_indices_full = indices_processing[fixed_indices_subset]

            if len(fixed_indices_full) > 0:
                ProjectedCov[fixed_indices_full] = Arem[fixed_indices_full]
                RevCholCov[fixed_indices_full] = subset_ReChCov[fixed_indices_subset]

            # Prepare for next iteration
            # NPD mask should be updated.
            # Those that failed (subset_NPD is True) remain in NPD mask.
            # Those that passed (subset_NPD is False) are removed from NPD mask.

            # We can just update NPD mask at indices_processing with subset_NPD
            NPD[indices_processing] = subset_NPD

            idxNPD = NPD # Update idxNPD for next loop

            if not np.any(idxNPD):
                break

        # Finalize NPD remediation processing and issue warnings
        if np.any(NPD):
            # If no allowable clipping factor sufficiently remediates
            Arem[NPD] = np.nan
            RevCholCov[NPD] = np.nan
            if WarningLevel > 0:
                warnings.warn(f"{np.sum(NPD)} non-positive definite covariances detected; unable to be remediated",
                              UserWarning)

        if WarningLevel > 2 and numFixed[0] > 0:
             warnings.warn(f"{numFixed[0]} non-positive definite covariance detected; remediated with standard clipping factor = {Fclip[0]}",
                           UserWarning)

        if WarningLevel > 1:
            for nclip in range(1, Nclip):
                if numFixed[nclip] > 0:
                    warnings.warn(f"{numFixed[nclip]} non-positive definite covariance detected; remediated with non-standard clipping factor = {Fclip[nclip]}",
                                  UserWarning)

    return Arem, RevCholCov, IsPosDef, IsRemediated


def _RevChol2x2(In):
    """
    Vectorized routine to perform reverse Cholesky factorization
    (equivalent to inv(chol(inv(X))))
    Returns [a, b, c] corresponding to lower triangular matrix elements.
    Or maybe it returns something else?
    MATLAB: Out = [a b c];
    a = sqrt(In(:,1)-b.^2);
    b = In(:,2)./c;
    c = sqrt(In(:,3));
    """
    # In is (N, 3)
    # c = sqrt(In(:,2)) -> index 2 (2,2)
    # b = In(:,1)./c    -> index 1 (1,2)
    # a = sqrt(In(:,0)-b.^2) -> index 0 (1,1)

    # Check MATLAB again:
    # c = sqrt(In(:,3)); -> In(:,3) is (2,2)
    # b = In(:,2)./c;    -> In(:,2) is (1,2)
    # a = sqrt(In(:,1)-b.^2); -> In(:,1) is (1,1)

    # Python indices: 0, 1, 2

    # Handling complex results: numpy sqrt of negative gives nan (warnings) or error?
    # Use explicit complex type if needed?
    # MATLAB handles sqrt(-1) as 1i automatically.
    # Python numpy defaults to warning and nan for negative real input unless complex type is used.
    # We need to detect imaginary components. So we should cast to complex.

    In_complex = In.astype(complex)

    c = np.sqrt(In_complex[:, 2])

    # Handle division by zero if c is 0?
    # In MATLAB 0/0 is NaN.
    # Python:
    with np.errstate(divide='ignore', invalid='ignore'):
        b = In_complex[:, 1] / c
        # If c is 0, b is inf or nan.

    a = np.sqrt(In_complex[:, 0] - b**2)

    # Stack a, b, c
    Out = np.column_stack((a, b, c))
    return Out

def _PosDef2x2(A):
    """
    Vectorized routine to perform a positive definite check using a Cholesky
    factorization on a 2x2 symmetric matrix.
    Returns:
        0: Positive Definite
        1: Failed 1st pivot test (A[0,0] <= 0)
        2: Failed 2nd pivot test
    """
    numElems = A.shape[0]

    # Check MATLAB:
    # ind1 = A(:,1) > 0; -> A(:,0) > 0
    ind1 = A[:, 0] > 0

    root = np.full(numElems, np.nan)
    root[ind1] = np.sqrt(A[ind1, 0])

    tempVal = np.full(numElems, np.nan)
    tempVal[ind1] = A[ind1, 1] / root[ind1]

    # ind2 = A(:,3) - tempVal(:).*tempVal(:) > 0; -> A(:,2)
    # Note: if tempVal is nan, comparison with > 0 is False.
    # But we want to check logic carefully.

    # Calculate A[2] - tempVal^2
    # If ind1 is False, tempVal is NaN, so this comparison is False (ind2 False).
    val2 = A[:, 2] - tempVal**2
    ind2 = val2 > 0

    posDef = np.full(numElems, np.nan)

    # posDef(~ind2) = 2; % Failed 2nd pivot test
    # posDef(~ind1) = 1; % Failed 1st pivot test
    # posDef(ind2) = 0; % If ind2 is true, then the 2x2 is positive definite

    # In python:
    # Initialize with something.
    posDef[:] = 2 # Default fail 2nd
    posDef[~ind1] = 1 # Fail 1st overrides
    posDef[ind2] = 0 # Pass overrides

    return posDef
