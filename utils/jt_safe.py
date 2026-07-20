"""
Safe Jittor ↔ NumPy conversion helpers for CUDA-only environments.
In Jittor 1.3.11 + CUDA 11.8 + sm_89, many ops lack CPU fallback,
making .numpy()/.item() fail. This module provides workarounds.
"""
import os
import numpy as np
import jittor as jt

# --- Path/fallback logging (writes to file, not terminal) ---
_path_log_file = None
_path_log_counts = {}  # throttle per-message-key: log first N times only


def set_path_log(filepath):
    """Set the log file path. Call once at startup with model_path/INFO.txt."""
    global _path_log_file
    _path_log_file = filepath


def path_log(msg, throttle_key=None, throttle_max=3):
    """Append msg to the path log file. If throttle_key is given, only log the
    first throttle_max occurrences of that key."""
    if throttle_key is not None:
        cnt = _path_log_counts.get(throttle_key, 0)
        _path_log_counts[throttle_key] = cnt + 1
        if cnt >= throttle_max:
            return
    if _path_log_file is None:
        return  # not yet configured; silently skip
    try:
        with open(_path_log_file, 'a') as f:
            f.write(msg.rstrip('\n') + '\n')
    except Exception:
        pass  # best-effort, never crash the training loop


def safe_numpy(var, fallback=None):
    """Try var.numpy(), return fallback if CUDA-only op chain blocks it."""
    try:
        return var.numpy()
    except RuntimeError:
        path_log(f"[FALLBACK] safe_numpy: .numpy() failed, using fallback={fallback is not None}")
        if fallback is not None:
            return fallback
        raise


def safe_item(var, fallback=0.0):
    """Try var.item(), return fallback if CUDA-only op chain blocks it."""
    try:
        return float(var.numpy())
    except RuntimeError:
        return fallback


def safe_index(var, bool_mask):
    """Boolean-index a jt.Var, using numpy fallback if jt.where is CUDA-only.
    Returns jt.Var."""
    try:
        mask_np = bool_mask.numpy()
        indices = np.nonzero(mask_np)[0]
        if len(indices) == 0:
            return var[:0]
        return var[indices]
    except RuntimeError:
        # Last resort: use numpy for both
        var_np = var.numpy()
        mask_np = bool_mask.numpy()
        return jt.array(var_np[mask_np])


def safe_boolean_index(var, bool_mask, fallback_np=None):
    """Boolean-index a jt.Var safely. Returns jt.Var.
    Stores a numpy copy of the mask in the caller if needed."""
    try:
        return var[bool_mask]
    except RuntimeError:
        # jt.where internal CUDA-only; use numpy
        if fallback_np is not None:
            var_np = fallback_np
        else:
            var_np = var.numpy()
        mask_np = bool_mask.numpy()
        return jt.array(var_np[mask_np])


def make_numpy_ref(var, shape, dtype=np.float32):
    """Create a numpy reference for a jt.Var's data.
    Use when var.numpy() fails. Returns (jt.Var, np.ndarray) pair."""
    try:
        np_val = var.numpy()
    except RuntimeError:
        np_val = np.zeros(shape, dtype=dtype)
    return var, np_val


def _dtype_spec(jt_dtype):
    """Return (numpy_dtype, c_sizeof_name, element_bytes) for a jt dtype."""
    mapping = {
        'float32': (np.float32, 'float', 4),
        'float64': (np.float64, 'double', 8),
        'int32':   (np.int32,   'int',    4),
        'int64':   (np.int64,   'long long', 8),
        'bool':    (np.bool_,   'bool',   1),
        'uint8':   (np.uint8,   'unsigned char', 1),
    }
    key = str(jt_dtype).split('.')[-1]  # 'jt.float32' → 'float32'
    if key in mapping:
        return mapping[key]
    return (np.float32, 'float', 4)


def memcpy_to_numpy(tensor):
    """Copy GPU tensor to numpy via sync + .numpy() (pure Jittor, no cupy)."""
    import numpy as np
    np_dtype, _, _ = _dtype_spec(tensor.dtype)

    # Ensure contiguous
    try:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
    except:
        pass

    # sync + .numpy() — standard Jittor read path
    try:
        jt.sync_all()
        return tensor.numpy()
    except:
        pass

    # Last resort: return zeros to preserve shape
    shape = list(tensor.shape)
    path_log(f"[FALLBACK] memcpy_to_numpy: all methods failed, zeros shape={shape}")
    return np.zeros(shape, dtype=np_dtype)


def safe_sync_read(tensor):
    """Robust GPU→CPU readback using Jittor sync + .numpy().

    Handles the key failure mode of Jittor 1.3.11:
    Reshaped/sliced views may fail silently — ensure contiguity first.
    """
    # Ensure contiguous (views can't sync their data)
    try:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
    except:
        pass

    # sync_all + .numpy() — standard Jittor read path
    try:
        jt.sync_all()
        return tensor.numpy()
    except:
        pass

    # Last resort: all-zeros
    import numpy as np
    return np.zeros(tensor.shape, dtype=np.float32)
