import numpy as np
import re
import os

def parse_value(value_str):
    """Parse a value string from the debug log into a Python/Numpy object."""
    value_str = value_str.strip()

    # Check for boolean
    if value_str == 'True': return True
    if value_str == 'False': return False

    # Check for None
    if value_str == 'None': return None

    # Check for dictionary
    if value_str.startswith('{'):
        # This is a bit risky but assuming simple dicts
        # We might need to handle 'np.float64(1.0)' inside dicts
        # Regex to replace 'np.float64(x)' with 'x'
        value_str = re.sub(r"np\.float64\((.*?)\)", r"\1", value_str)
        # Regex to handle 'array([...])' - difficult.
        # If dict contains arrays, we might skip parsing it fully or just eval if safe.
        # For now, let's try eval with numpy context
        try:
            return eval(value_str, {"np": np, "array": np.array, "float64": np.float64, "inf": np.inf, "nan": np.nan})
        except:
            return value_str # Return as string if eval fails

    # Check for array
    if value_str.startswith('['):
        # Clean up the string to be numpy friendly
        # Remove multiple newlines and spaces
        cleaned = value_str.replace('\n', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.replace('[', '').replace(']', '')
        # Convert to numpy array
        if cleaned.strip() == '': return np.array([])
        arr = np.fromstring(cleaned, sep=' ')

        # Try to infer shape from the original string's brackets
        # Count opening brackets at the start
        dims = 0
        for char in value_str:
            if char == '[': dims += 1
            else: break

        if dims > 1:
            # It's multi-dimensional.
            # For 2D, we can count rows based on ']\n [' sequence?
            # Or assume 6x6 if 36 elements, 3x1 if 3 elements, etc.
            if arr.size == 36:
                return arr.reshape(6, 6)
            elif arr.size == 9:
                return arr.reshape(3, 3) # or (1,9) or (9,1)
            elif arr.size == 3:
                return arr.reshape(3, 1) # or (1,3) or (3,)
            elif arr.size == 6:
                return arr.reshape(6, 1) # or (1,6)
            # Add more heuristics as needed

        return arr

    # Check for number
    try:
        if '.' in value_str or 'e' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except:
        pass

    return value_str # Return as string

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    sections = {}

    # Split by "=== Label ==="
    # Regex: === (.*?) ===\n(.*?)=== End \1 ===
    # DOTALL to match newlines
    matches = re.finditer(r"=== (.*?) ===\n(.*?)=== End \1 ===", content, re.DOTALL)

    for match in matches:
        label = match.group(1)
        body = match.group(2)

        variables = {}
        # Split variables by "--- Name ---"
        # Since variables are sequential, we can use finditer again
        # But we need to capture the text between headers.

        var_matches = list(re.finditer(r"--- (.*?) ---\n", body))
        for i, v_match in enumerate(var_matches):
            var_name = v_match.group(1)
            start_idx = v_match.end()
            if i + 1 < len(var_matches):
                end_idx = var_matches[i+1].start()
            else:
                end_idx = len(body)

            val_str = body[start_idx:end_idx].strip()
            variables[var_name] = parse_value(val_str)

        sections[label] = variables

    return sections

def generate_test_file(sections, output_path):
    with open(output_path, 'w') as f:
        f.write('"""Unit tests generated from Pc3D_Hall_debug_py.txt instrumentation log."""\n\n')
        f.write('import unittest\n')
        f.write('import numpy as np\n')
        f.write('from DistributedPython.ProbabilityOfCollision.Utils.EquinoctialMatrices import EquinoctialMatrices\n')
        f.write('from DistributedPython.ProbabilityOfCollision.Utils.conj_bounds_Coppola import conj_bounds_Coppola\n\n')

        f.write('class TestPc3DHallDebugData(unittest.TestCase):\n\n')

        # Test EquinoctialMatrices Primary
        if 'Inputs: EquinoctialMatrices (Primary)' in sections and 'Outputs: EquinoctialMatrices (Primary)' in sections:
            f.write('    def test_EquinoctialMatrices_Primary(self):\n')
            inputs = sections['Inputs: EquinoctialMatrices (Primary)']
            outputs = sections['Outputs: EquinoctialMatrices (Primary)']

            write_assignment(f, 'r1', inputs['r1'], indent=8)
            write_assignment(f, 'v1', inputs['v1'], indent=8)
            write_assignment(f, 'C1', inputs['C1'], indent=8)
            write_assignment(f, 'rem_flag', inputs['rem_flag'], indent=8)

            f.write('\n        # Run function\n')
            f.write('        X, P, E, J, K, Q, QStat, QRaw, QRem, CRem = EquinoctialMatrices(r1.flatten(), v1.flatten(), C1, rem_flag)\n\n')

            f.write('        # Verify outputs\n')
            for key, val in outputs.items():
                target_var = map_output_name_to_var(key)
                if target_var:
                    write_assertion(f, target_var, val, indent=8)
            f.write('\n')

        # Test EquinoctialMatrices Secondary
        if 'Inputs: EquinoctialMatrices (Secondary)' in sections and 'Outputs: EquinoctialMatrices (Secondary)' in sections:
            f.write('    def test_EquinoctialMatrices_Secondary(self):\n')
            inputs = sections['Inputs: EquinoctialMatrices (Secondary)']
            outputs = sections['Outputs: EquinoctialMatrices (Secondary)']

            write_assignment(f, 'r2', inputs['r2'], indent=8)
            write_assignment(f, 'v2', inputs['v2'], indent=8)
            write_assignment(f, 'C2', inputs['C2'], indent=8)
            write_assignment(f, 'rem_flag', inputs['rem_flag'], indent=8)

            f.write('\n        # Run function\n')
            f.write('        X, P, E, J, K, Q, QStat, QRaw, QRem, CRem = EquinoctialMatrices(r2.flatten(), v2.flatten(), C2, rem_flag)\n\n')

            f.write('        # Verify outputs\n')
            for key, val in outputs.items():
                target_var = map_output_name_to_var(key)
                if target_var:
                    write_assertion(f, target_var, val, indent=8)
            f.write('\n')

        # Test conj_bounds_Coppola
        if 'Inputs: conj_bounds_Coppola' in sections and 'Outputs: conj_bounds_Coppola' in sections:
            f.write('    def test_conj_bounds_Coppola(self):\n')
            inputs = sections['Inputs: conj_bounds_Coppola']
            outputs = sections['Outputs: conj_bounds_Coppola']

            write_assignment(f, 'gamma', inputs['gamma'], indent=8)
            write_assignment(f, 'HBR', inputs['HBR'], indent=8)
            write_assignment(f, 'r', inputs['r'], indent=8)
            write_assignment(f, 'v', inputs['v'], indent=8)
            write_assignment(f, 'C', inputs['C'], indent=8)

            f.write('\n        # Run function\n')
            # Assuming verbose=False for test
            f.write('        tau0, tau1, tau0_gam1, tau1_gam1 = conj_bounds_Coppola(gamma, HBR, r, v, C, False)\n\n')

            f.write('        # Verify outputs\n')
            write_assertion(f, 'tau0', outputs['tau0'], indent=8)
            write_assertion(f, 'tau1', outputs['tau1'], indent=8)
            write_assertion(f, 'tau0_gam1', outputs['tau0_gam1'], indent=8)
            write_assertion(f, 'tau1_gam1', outputs['tau1_gam1'], indent=8)
            f.write('\n')

        f.write('if __name__ == "__main__":\n')
        f.write('    unittest.main()\n')

def map_output_name_to_var(name):
    """Map log variable names to return tuple variables for EquinoctialMatrices."""
    mapping = {
        'Xmean10': 'X', 'Pmean10': 'P', 'Emean10': 'E', 'Jmean10': 'J', 'Kmean10': 'K',
        'Qmean10': 'Q', 'Qmean10RemStat': 'QStat', 'Qmean10Raw': 'QRaw', 'Qmean10Rem': 'QRem', 'C1Rem': 'CRem',
        'Xmean20': 'X', 'Pmean20': 'P', 'Emean20': 'E', 'Jmean20': 'J', 'Kmean20': 'K',
        'Qmean20': 'Q', 'Qmean20RemStat': 'QStat', 'Qmean20Raw': 'QRaw', 'Qmean20Rem': 'QRem', 'C2Rem': 'CRem'
    }
    return mapping.get(name)

def write_assignment(f, name, value, indent):
    ind = ' ' * indent
    if isinstance(value, np.ndarray):
        # Format array nicely
        # Use repr, but replace 'array(' with 'np.array('
        s = np.array2string(value, separator=', ', threshold=np.inf, max_line_width=np.inf)
        # Add np.array wrap
        s = f"np.array({s})"
        # If it was reshaped in parsing, ensure we keep that shape.
        # But array2string prints brackets reflecting shape.
        f.write(f"{ind}{name} = {s}\n")
    else:
        f.write(f"{ind}{name} = {repr(value)}\n")

def write_assertion(f, name, expected, indent):
    ind = ' ' * indent
    if isinstance(expected, np.ndarray):
        s = np.array2string(expected, separator=', ', threshold=np.inf, max_line_width=np.inf)
        f.write(f"{ind}expected_{name} = np.array({s})\n")
        f.write(f"{ind}np.testing.assert_allclose({name}, expected_{name}, rtol=1e-8, atol=1e-8)\n")
    else:
        f.write(f"{ind}self.assertAlmostEqual({name}, {expected}, places=8)\n")

if __name__ == "__main__":
    filepath = 'Pc3D_Hall_debug_py.txt'
    if os.path.exists(filepath):
        sections = parse_log_file(filepath)
        generate_test_file(sections, 'DistributedPython/ProbabilityOfCollision/tests/test_Pc3D_Hall_debug_data.py')
        print("Test file generated.")
    else:
        print(f"File {filepath} not found.")
