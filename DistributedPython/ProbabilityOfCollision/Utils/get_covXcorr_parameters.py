import numpy as np

def get_covXcorr_parameters(params):
    """
    Gets covariance cross-correlation parameters from the parameter structure passed in.

    Args:
        params (dict): Dictionary with the following fields:
            params['covXcorr'] (dict): Dictionary with the following fields:
                params['covXcorr']['sigp']: DCP sigma for the primary object
                params['covXcorr']['sigs']: DCP sigma for the secondary object
                params['covXcorr']['Gvecp']: 1x6 DCP sensitivity vector for the primary object
                params['covXcorr']['Gvecs']: 1x6 DCP sensitivity vector for the secondary object

    Returns:
        tuple:
            covXcorr (bool): true/false indicator of whether or not cross-correlation
                             information was correctly read
            sigp (float): DCP sigma for the primary object
            Gvecp (np.ndarray): 1x6 DCP sensitivity vector for the primary object
            sigs (float): DCP sigma for the secondary object
            Gvecs (np.ndarray): 1x6 DCP sensitivity vector for the secondary object
    """

    # Initialize cross correlation processing flag to false, and only change to
    # true if valid covXcorr parameters are found in parameters structure
    covXcorr = False

    # Initialize output DCP sigma values and sensitivity vectors
    sigp = None
    Gvecp = None
    sigs = None
    Gvecs = None

    # Check for valid covXcorr parameters
    if params is not None and 'covXcorr' in params and params['covXcorr']:
        cov_params = params['covXcorr']

        # Extract DCP sigma values
        sigp = cov_params.get('sigp')
        sigs = cov_params.get('sigs')

        # Extract DCP sensitivity vectors
        Gvecp = cov_params.get('Gvecp')
        Gvecs = cov_params.get('Gvecs')

        # Return with false covXcorr flag if any DCP quantities are empty/None
        if (sigp is None or Gvecp is None or
            sigs is None or Gvecs is None):
            return covXcorr, sigp, Gvecp, sigs, Gvecs

        # Ensure Gvecp/Gvecs are numpy arrays
        Gvecp = np.asarray(Gvecp)
        Gvecs = np.asarray(Gvecs)

        # Check for correct dimensions
        # sigp and sigs should be scalar
        if np.ndim(sigp) != 0 or np.ndim(sigs) != 0:
             raise ValueError('Incorrect DCP value dimensions')

        # Gvecp and Gvecs should be 1x6.
        if Gvecp.shape != (1, 6):
             raise ValueError('Incorrect DCP value dimensions')
        if Gvecs.shape != (1, 6):
             raise ValueError('Incorrect DCP value dimensions')

        # Check for invalid DCP values
        if np.isnan(sigp) or sigp < 0 or np.isnan(sigs) or sigs < 0:
            raise ValueError('Invalid DCP sigma value(s) found')

        if np.any(np.isnan(Gvecp)) or np.any(np.isnan(Gvecs)):
            raise ValueError('Invalid DCP sensitivity vector value(s) found')

        # At this point, all checks have been passed so set the covariance
        # cross correlation processing flag to true
        covXcorr = True

    return covXcorr, sigp, Gvecp, sigs, Gvecs
