"""Historical-response numerical estimators."""
from pathlib import Path
import hashlib,json,resource,signal,sys,time
import numpy as np
import pandas as pd

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def require_year(frame, year):
    years = pd.to_datetime(frame.date).dt.year if 'date' in frame else frame.year
    if not years.eq(year).all():
        raise ValueError(f'Expected history year {year}, got {sorted(years.unique())}')

def fit_local_response(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite calibration values')
    keep = x != 0
    if not keep.any():
        return 0.0
    ratios = y[keep] / x[keep]
    weights = np.abs(x[keep])
    order = np.argsort(ratios, kind='stable')
    k = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side='left')
    return float(ratios[order[min(k, len(order) - 1)]])

def fit_within_response(x, y, groups, alpha=1.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    xc = x.copy()
    yc = y.copy()
    for group in np.unique(groups):
        mask = groups == group
        xc[mask] -= x[mask].mean(axis=0)
        yc[mask] -= y[mask].mean()
    return np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)

def feature_matrix(spec, frame):
    from scipy.interpolate import BSpline
    blocks = []
    for transform in spec['transformers']:
        cols = transform['input_columns']
        x = frame[cols].to_numpy(float)
        if transform['class'] == 'passthrough':
            blocks.append(x)
        else:
            basis = transform['learned_bspline_basis'][0]
            n = basis['coefficient_matrix_shape'][0]
            bs = BSpline(basis['knot_vector'], np.eye(n), basis['degree'], extrapolate=True)
            blocks.append(bs(x[:, 0])[:, :-1])
    x = np.hstack(blocks) / np.asarray(spec['standard_scaler']['scale'])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite feature matrix')
    return x

class RunGuard:

    def __init__(self, paper):
        self.paper = paper
        self.start = time.monotonic()
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Registered 3600-second limit')))
        signal.alarm(3600)

    def check(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        if rss > 4 * 1024 ** 3:
            raise MemoryError('Registered 4-GiB limit exceeded')
        return dict(wall_seconds=time.monotonic() - self.start, peak_rss_bytes=rss, wall_limit_seconds=3600, memory_limit_bytes=4 * 1024 ** 3)

    def finish(self):
        out = self.check()
        signal.alarm(0)
        return out
