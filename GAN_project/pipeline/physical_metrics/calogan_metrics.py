import numpy as np


def _sum_layers_if_needed(data):
    """
    Приводит данные к виду (batch, rows, cols).

    Ожидаемый формат для нового датасета:
    - 4D: (batch, layers, rows, cols) = (batch, 7, 7, 5)
    - 3D: (batch, rows, cols) = (batch, 7, 5)
    """
    if data.ndim == 4:
        data = np.sum(data, axis=1)
    return data


def _safe_divide(num, den, eps=1e-8):
    """
    Безопасное деление без RuntimeWarning divide by zero.
    """
    return np.divide(
        num,
        den,
        out=np.zeros_like(num, dtype=float),
        where=np.abs(den) > eps
    )


def _make_grid(batch_size, rows, cols):
    """
    Создает координатную сетку размера rows x cols.
    Для твоего случая rows=7, cols=5.
    """
    x = np.linspace(-14.5, 14.5, cols)
    y = np.linspace(-14.5, 14.5, rows)

    xx, yy = np.meshgrid(x, y)

    xx = np.repeat(xx[np.newaxis, ...], batch_size, axis=0)
    yy = np.repeat(yy[np.newaxis, ...], batch_size, axis=0)

    return xx, yy


def get_assymetry(imgs, ps, points, orthog=False):
    # (batch, 7, 7, 5) -> (batch, 7, 5)
    imgs = _sum_layers_if_needed(imgs)

    batch_size, rows, cols = imgs.shape
    zoff = 25

    xx, yy = _make_grid(batch_size, rows, cols)

    # безопасно делим на pz
    px_over_pz = _safe_divide(ps[:, 0], ps[:, 2])
    py_over_pz = _safe_divide(ps[:, 1], ps[:, 2])

    points_0 = points[:, 0] + zoff * px_over_pz
    points_1 = points[:, 1] + zoff * py_over_pz

    px_over_py = _safe_divide(ps[:, 0], ps[:, 1])
    py_over_px = _safe_divide(ps[:, 1], ps[:, 0])

    if orthog:
        line_func = lambda x: (
            (x - points_0[..., np.newaxis, np.newaxis])
            / (px_over_py[..., np.newaxis, np.newaxis] + 1e-8)
            + points_1[..., np.newaxis, np.newaxis]
        )
    else:
        line_func = lambda x: (
            -(x - points_0[..., np.newaxis, np.newaxis])
            / (py_over_px[..., np.newaxis, np.newaxis] + 1e-8)
            + points_1[..., np.newaxis, np.newaxis]
        )

    sign = np.ones(batch_size)
    if not orthog:
        sign = (ps[:, 1] > 0).astype(int)
        sign = 2 * (sign - 0.5)

    idx = np.where((yy - line_func(xx)) * sign[..., np.newaxis, np.newaxis] < 0)

    zz = np.ones((batch_size, rows, cols))
    zz[idx] = 0

    denom = np.sum(imgs, axis=(1, 2))
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    assym = (
        np.sum(imgs * zz, axis=(1, 2))
        - np.sum(imgs * (1 - zz), axis=(1, 2))
    ) / denom

    return np.nan_to_num(assym)


def zz_to_line(zz):
    batch_size, rows, cols = zz.shape

    res = (
        np.concatenate(
            [np.abs(np.diff(zz, axis=2)), np.zeros((batch_size, rows, 1))],
            axis=2
        )
        +
        np.concatenate(
            [np.abs(np.diff(zz, axis=1)), np.zeros((batch_size, 1, cols))],
            axis=1
        )
    )

    return np.clip(res, 0, 1)


def get_shower_width(data, ps, points, orthog=False):
    # (batch, 7, 7, 5) -> (batch, 7, 5)
    data = _sum_layers_if_needed(data)

    batch_size, rows, cols = data.shape
    zoff = 25

    xx, yy = _make_grid(batch_size, rows, cols)

    # безопасно делим на pz
    px_over_pz = _safe_divide(ps[:, 0], ps[:, 2])
    py_over_pz = _safe_divide(ps[:, 1], ps[:, 2])

    points_0 = points[:, 0] + zoff * px_over_pz
    points_1 = points[:, 1] + zoff * py_over_pz

    px_over_py = _safe_divide(ps[:, 0], ps[:, 1])
    py_over_px = _safe_divide(ps[:, 1], ps[:, 0])

    if orthog:
        line_func = lambda x: (
            -(x - points_0[..., np.newaxis, np.newaxis])
            / (px_over_py[..., np.newaxis, np.newaxis] + 1e-8)
            + points_1[..., np.newaxis, np.newaxis]
        )
    else:
        line_func = lambda x: (
            (x - points_0[..., np.newaxis, np.newaxis])
            / (py_over_px[..., np.newaxis, np.newaxis] + 1e-8)
            + points_1[..., np.newaxis, np.newaxis]
        )

    rescale = np.sqrt(1 + py_over_px**2)

    sign = np.ones(batch_size)
    if not orthog:
        sign = (ps[:, 1] < 0).astype(int)
        sign = 2 * (sign - 0.5)

    idx = np.where((yy - line_func(xx)) * sign[..., np.newaxis, np.newaxis] < 0)

    zz = np.ones((batch_size, rows, cols))
    zz[idx] = 0

    line = zz_to_line(zz)

    ww = line * data

    sum_0 = ww.sum(axis=(1, 2))
    sum_1 = (ww * rescale[..., np.newaxis, np.newaxis] * xx).sum(axis=(1, 2))
    sum_2 = (ww * (rescale[..., np.newaxis, np.newaxis] * xx) ** 2).sum(axis=(1, 2))

    sum_1 = sum_1 / (sum_0 + 1e-5)
    sum_2 = sum_2 / (sum_0 + 1e-5)

    sigma = np.sqrt(np.maximum(sum_2 - sum_1 * sum_1, 0))

    return np.nan_to_num(sigma)


def get_ms_ratio2(data, alpha=0.1):
    # (batch, 7, 7, 5) -> (batch, 7, 5)
    data = _sum_layers_if_needed(data)

    ms = np.sum(data, axis=(1, 2))
    num = np.sum(
        data >= (ms * alpha)[:, np.newaxis, np.newaxis],
        axis=(1, 2)
    )

    # Новый размер центральной матрицы: 7 x 5 = 35
    return num / (data.shape[1] * data.shape[2])


def get_sparsity_level(data):
    # (batch, 7, 7, 5) -> (batch, 7, 5)
    data = _sum_layers_if_needed(data)

    alphas = np.logspace(-5, -1, 20)
    sparsity = []

    for alpha in alphas:
        sparsity.append(get_ms_ratio2(data, alpha))

    return np.array(sparsity)
