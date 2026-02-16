def set_default_param(params, param_name, default_param_value):
    """
    Set a default value in a params dictionary.

    Args:
        params (dict): The parameters dictionary. If None, a new dictionary is created.
        param_name (str): The name of the parameter to set.
        default_param_value (any): The default value to set if the parameter is missing or None.

    Returns:
        dict: The updated parameters dictionary.
    """
    if not isinstance(param_name, str) or not param_name:
        raise ValueError("Invalid input param_name")

    if params is None:
        params = {}

    if param_name not in params:
        params[param_name] = default_param_value
    elif params[param_name] is None and default_param_value is not None:
        params[param_name] = default_param_value

    return params
