run_example:
	octave --no-gui --eval "addpath('DistributedMatlab/ProbabilityOfCollision'); addpath('DistributedMatlab/ProbabilityOfCollision/Pc3D_Hall_Utils'); run_Pc3D_Hall_Example1_debug;"

run_example_py:
	export PYTHONPATH=. && python3 DistributedPython/ProbabilityOfCollision/Pc3D_Hall_Utils/run_Pc3D_Hall_Example1_debug.py
