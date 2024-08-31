import numpy as np
import warnings

def bootstrap(*args, **kwargs):
    """Resample one or more arrays with replacement and compute aggregate values.

    Parameters
    ----------
    *args : array_like
        Arrays to bootstrap along the first axis.
    n_boot : int, default=10000
        Number of bootstrap iterations.
    axis : int, default=None
        Axis to pass to `func` as a keyword argument.
    units : array_like, default=None
        Array of sampling unit IDs. Resample units and then observations within units if provided.
    func : str or callable, default="mean"
        Function to apply to the resampled arrays. If string, it uses the corresponding numpy function.
    seed : int, np.random.Generator, np.random.RandomState, or None, default=None
        Seed for the random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of bootstrapped statistic values.

    Raises
    ------
    ValueError
        If input arrays are not of the same length.
    """
    # Ensure all input arrays are the same length
    array_lengths = list(map(len, args))
    if len(set(array_lengths)) > 1:
        raise ValueError("All input arrays must have the same length")
    n = array_lengths[0]

    # Default keyword arguments
    n_boot = kwargs.get("n_boot", 10000)
    func = kwargs.get("func", "mean")
    axis = kwargs.get("axis", None)
    units = kwargs.get("units", None)
    seed = kwargs.get("seed", None)

    # Initialize the random number generator
    rng = np.random.default_rng(seed)

    # Convert inputs to numpy arrays
    args = [np.asarray(arg) for arg in args]
    if units is not None:
        units = np.asarray(units)

    # Determine the function to apply
    if isinstance(func, str):
        try:
            f = getattr(np, func)
        except AttributeError:
            raise ValueError(f"Function `{func}` not found in numpy")
        
        # Check for NaN-aware function
        if np.isnan(np.sum(np.column_stack(args))) and not func.startswith("nan"):
            nan_func = getattr(np, f"nan{func}", None)
            if nan_func is None:
                warnings.warn(f"Data contain NaNs but no nan-aware version of `{func}` found", UserWarning)
            else:
                f = nan_func
    else:
        f = func

    # Perform the bootstrap
    if units is not None:
        return _structured_bootstrap(args, n_boot, units, f, {"axis": axis}, rng.integers)

    boot_dist = np.empty(n_boot)
    for i in range(n_boot):
        resampler = rng.integers(0, n, n)
        sample = [a[resampler] for a in args]
        boot_dist[i] = f(*sample, axis=axis)
    return boot_dist

def _structured_bootstrap(args, n_boot, units, func, func_kwargs, integers):
    """Resample units instead of individual datapoints."""
    unique_units = np.unique(units)
    n_units = len(unique_units)

    # Organize data by units
    args_by_unit = [[a[units == unit] for unit in unique_units] for a in args]

    boot_dist = np.empty(n_boot)
    for i in range(n_boot):
        unit_resampler = integers(0, n_units, n_units)
        sampled_units = [args_by_unit[j][unit_resampler] for j in range(len(args))]
        sampled_lengths = [len(sample) for sample in sampled_units[0]]
        
        # Resample observations within units
        resampled_data = []
        for sample, length in zip(sampled_units, sampled_lengths):
            resample_idx = [integers(0, length, length) for _ in range(len(sample))]
            resampled_data.append([unit.take(idx, axis=0) for unit, idx in zip(sample, resample_idx)])
        
        # Flatten and concatenate
        flat_sample = [np.concatenate(data) for data in zip(*resampled_data)]
        boot_dist[i] = func(*flat_sample, **func_kwargs)
    
    return boot_dist
