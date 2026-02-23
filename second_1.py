import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

PI = math.pi

# Корни z^3 = 1
ROOTS = np.array([
    1.0 + 0.0j,
    cmath.rect(1.0, 2.0 * PI / 3.0),
    cmath.rect(1.0, 4.0 * PI / 3.0),
], dtype=np.complex128)


def newton_fractal(xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0,
                   nx=800, ny=800, tol=1e-12, max_iter=50):
    x = np.linspace(xmin, xmax, nx, dtype=np.float64)
    y = np.linspace(ymin, ymax, ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    basin = np.full(Z.shape, -1, dtype=np.int32)
    iters = np.zeros(Z.shape, dtype=np.int32)

    active = np.ones(Z.shape, dtype=bool)

    for k in range(1, max_iter + 1):
        Za = Z[active]
        fz = Za**3 - 1.0

        done = np.abs(fz) < tol
        if np.any(done):
            idx_active = np.argwhere(active)
            idx_done = idx_active[done]

            Z_done = Za[done]
            d = np.abs(Z_done[:, None] - ROOTS[None, :]
            cls = np.argmin(d, axis=1).astype(np.int32)

            basin[idx_done[:, 0], idx_done[:, 1]] = cls
            iters[idx_done[:, 0], idx_done[:, 1]] = k

            active[idx_done[:, 0], idx_done[:, 1]] = False

        if not np.any(active):
            break

        Za = Z[active]
        dfz = 3.0 * Za**2
        good = np.abs(dfz) > 1e-14

        Z_new = Za.copy()
        Z_new[good] = Za[good] - (Za[good]**3 - 1.0) / dfz[good]

        idx_active = np.argwhere(active)
        Z[idx_active[:, 0], idx_active[:, 1]] = Z_new

        if np.any(~good):
            idx_bad = idx_active[~good]
            active[idx_bad[:, 0], idx_bad[:, 1]] = False

    return basin, iters


def save_png(basin, iters, xmin, xmax, ymin, ymax, out_path="newton_z3_minus_1.png"):
    colors = np.array([
        [1.0, 0.2, 0.2],  # root 0
        [0.2, 1.0, 0.2],  # root 1
        [0.2, 0.2, 1.0],  # root 2
    ], dtype=np.float64)

    img = np.zeros((basin.shape[0], basin.shape[1], 3), dtype=np.float64)
    mask = basin >= 0
    img[mask] = colors[basin[mask]]

    # яркость по скорости: меньше итераций -> ярче
    it = iters.astype(np.float64)
    it[~mask] = np.nan
    max_it = np.nanmax(it) if np.any(mask) else 1.0
    shade = 1.0 - (it / max_it)
    shade = np.nan_to_num(shade, nan=0.0)

    img *= (0.35 + 0.65 * shade[..., None])
    img[~mask] = 0.0  # несошедшиеся — чёрные

    plt.figure(figsize=(7, 7), dpi=250)
    plt.imshow(img, origin="lower", extent=[xmin, xmax, ymin, ymax])
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("Newton fractal for z^3 - 1")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0

    basin, iters = newton_fractal(
        xmin, xmax, ymin, ymax,
        nx=800, ny=800,
        tol=1e-12,
        max_iter=50,
    )
    save_png(basin, iters, xmin, xmax, ymin, ymax, out_path="newton_z3_minus_1.png")
    print("Saved: newton_z3_minus_1.png")


if __name__ == "__main__":
    main()

