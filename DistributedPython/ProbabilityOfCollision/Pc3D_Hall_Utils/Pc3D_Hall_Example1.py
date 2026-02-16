import numpy as np
import sys
import os

# Add the root directory to sys.path to allow imports from DistributedPython
# Assuming this script is run from the repo root or its own directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from DistributedPython.ProbabilityOfCollision.PcElrod import PcElrod
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall import Pc3D_Hall

def main():
    # Example validation case for the Pc3D_Hall.m function, using data from an
    # actual NASA CARA conjunction.
    #
    # Case 1: Example with 3D-Pc and 2D-Pc nearly equal to one another (as is
    #         the case with the vast majority of high relative velocity
    #         conjunctions).
    #
    # Executing the test case code below should yield the following output:
    #
    #     Pc2D = 1.0281653e-02
    #     Pc3D = 1.0281834e-02
    #

    r1 = np.array([[-9.841950433215101e+05, +3.932342044549424e+05, +6.991223682230414e+06]])
    v1 = np.array([[+4.883696742000000e+03, +5.689086045000000e+03, +3.665361590000000e+02]])
    cov1 = np.array([
        [+4.976545641899520e+04, +5.787130862568278e+04, +3.370410320935015e+03, +1.137272273949272e+01, -4.325472616114674e+00, -8.009705480233521e+01],
        [+5.787130862568278e+04, +6.730377643610841e+04, +3.926542932121541e+03, +1.321992688238858e+01, -5.035560720747812e+00, -9.314985106902773e+01],
        [+3.370410320935015e+03, +3.926542932121541e+03, +2.461403197221289e+02, +7.586865834476763e-01, -3.077848629905763e-01, -5.434034460756914e+00],
        [+1.137272273949272e+01, +1.321992688238858e+01, +7.586865834476763e-01, +2.608186227148725e-03, -9.804181796720670e-04, -1.829751672999786e-02],
        [-4.325472616114674e+00, -5.035560720747812e+00, -3.077848629905763e-01, -9.804181796720670e-04, +3.895883508545853e-04, +6.968892326415779e-03],
        [-8.009705480233521e+01, -9.314985106902773e+01, -5.434034460756914e+00, -1.829751672999786e-02, +6.968892326415779e-03, +1.289253320300791e-01]
    ])

    r2 = np.array([[-9.839696058965517e+05, +3.936845951174244e+05, +6.991219291625473e+06]])
    v2 = np.array([[+1.509562687000000e+03, +7.372938617000000e+03, -1.492509430000000e+02]])
    cov2 = np.array([
        [+4.246862551076427e+04, +2.066374367781032e+05, -5.011108933888592e+03, +3.104606531932427e+01, -1.201093683199582e+01, -2.207975848324051e+02],
        [+2.066374367781032e+05, +1.005854717283451e+06, -2.434876491048039e+04, +1.510022508670080e+02, -5.850063541467530e+01, -1.074752763805685e+03],
        [-5.011108933888592e+03, -2.434876491048039e+04, +6.131274993037449e+02, -3.667147183233717e+00, +1.391769957262238e+00, +2.601457791444154e+01],
        [+3.104606531932427e+01, +1.510022508670080e+02, -3.667147183233717e+00, +2.272826228568773e-02, -8.778253314778023e-03, -1.613538091053610e-01],
        [-1.201093683199582e+01, -5.850063541467530e+01, +1.391769957262238e+00, -8.778253314778023e-03, +3.428801115804722e-03, +6.251148178133809e-02],
        [-2.207975848324051e+02, -1.074752763805685e+03, +2.601457791444154e+01, -1.613538091053610e-01, +6.251148178133809e-02, +1.148404222181769e+00]
    ])

    HBR = 20

    # PcElrod returns (Pc, Arem, IsPosDef, IsRemediated)
    # Using index [0] to get Pc
    Pc2D = PcElrod(r1, v1, cov1, r2, v2, cov2, HBR)[0]

    # Pc3D_Hall returns (Pc, out)
    Pc3D = Pc3D_Hall(r1, v1, cov1, r2, v2, cov2, HBR)[0]

    # Handle potential complex return type from PcElrod (although imag part should be 0)
    val_Pc2D = Pc2D[0] if Pc2D.size > 0 else 0.0
    if np.iscomplexobj(val_Pc2D):
        val_Pc2D = val_Pc2D.real

    # Note: Using %0.7e format as in the MATLAB example
    print(f"Pc2D = {val_Pc2D:0.7e}")
    print(f"Pc3D = {Pc3D:0.7e}")

if __name__ == '__main__':
    main()
