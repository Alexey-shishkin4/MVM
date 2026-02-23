import math
import cmath

PI = math.pi


# ============================================================
# 1) tan(x) = x  ->  f(x) = tan(x) - x
# ============================================================

def f(x: float) -> float:
    return math.tan(x) - x


def df(x: float) -> float:
    # f'(x) = sec^2(x) - 1 = tan^2(x)
    t = math.tan(x)
    return t * t


def bisection_method(a: float, b: float, ftol: float, xtol: float, max_iter: int) -> tuple[float, int, bool]:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        raise ValueError("Ошибка: f(a) и f(b) должны иметь разные знаки!")

    mid = (a + b) * 0.5
    for it in range(1, max_iter + 1):
        mid = (a + b) * 0.5
        fmid = f(mid)

        if abs(fmid) < ftol or (b - a) * 0.5 < xtol:
            return mid, it, True

        if fa * fmid <= 0.0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    return mid, max_iter, False


def fixed_point_iteration_atan(x0: float, ftol: float, xtol: float, max_iter: int) -> tuple[float, int, bool]:
    """
    tan(x)=x  <=>  x=atan(x)
    Итерация: x_{n+1} = atan(x_n)
    Может быть очень медленной около 0, поэтому max_iter держим большим. [web:130]
    """
    x = x0
    for it in range(1, max_iter + 1):
        x_next = math.atan(x)

        if abs(x_next - x) < xtol and abs(f(x_next)) < ftol:
            return x_next, it, True

        x = x_next

    return x, max_iter, False


def newton_method_modified(x0: float, ftol: float, xtol: float, max_iter: int, multiplicity: int = 1) -> tuple[float, int, bool]:
    """
    Модифицированный Ньютон для кратного корня:
      x_{n+1} = x_n - m * f(x_n)/f'(x_n)  [web:139]
    Для корня x=0 у tan(x)-x кратность 3 -> m=3. [web:139]
    """
    x = x0
    m = float(multiplicity)

    for it in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < ftol:
            return x, it, True

        dfx = df(x)
        if abs(dfx) < 1e-14:
            return x, it, False

        x_next = x - m * fx / dfx

        if abs(x_next - x) < xtol and abs(f(x_next)) < ftol:
            return x_next, it, True

        x = x_next

    return x, max_iter, False


def secant_method(x0: float, x1: float, ftol: float, xtol: float, max_iter: int) -> tuple[float, int, bool]:
    f0 = f(x0)
    f1 = f(x1)

    for it in range(1, max_iter + 1):
        denom = f1 - f0
        if abs(denom) < 1e-14:
            return x1, it, False

        x2 = x1 - f1 * (x1 - x0) / denom

        if abs(f(x2)) < ftol or abs(x2 - x1) < xtol:
            return x2, it, abs(f(x2)) < ftol

        x0, x1 = x1, x2
        f0, f1 = f1, f(x2)

    return x1, max_iter, False


# ============================================================
# 2) Complex Newton for z^3 - 1 = 0
# ============================================================

def F(z: complex) -> complex:
    return z**3 - 1.0


def dF(z: complex) -> complex:
    return 3.0 * (z**2)


def newton_complex(z0: complex, tol: float, max_iter: int) -> tuple[complex, int, bool]:
    z = z0
    for it in range(1, max_iter + 1):
        fz = F(z)
        if abs(fz) < tol:
            return z, it, True

        dfz = dF(z)
        if abs(dfz) < 1e-14:
            return z, it, False

        z_next = z - fz / dfz
        if abs(z_next - z) < tol and abs(F(z_next)) < tol:
            return z_next, it, True
        z = z_next

    return z, max_iter, False


def which_root(z: complex) -> int:
    roots = [
        1.0 + 0.0j,
        cmath.rect(1.0, 2.0 * PI / 3.0),
        cmath.rect(1.0, 4.0 * PI / 3.0),
    ]
    min_idx = 0
    min_dist = abs(z - roots[0])
    for i in range(1, len(roots)):
        dist = abs(z - roots[i])
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx


def main():
    ftol = 1e-8
    xtol = 1e-12

    print("=== Уравнение tan(x) = x ===")

    # 1) Бисекция: корень в [-1,1] — это 0, поэтому сойдётся мгновенно
    a, b = -1.0, 1.0
    root, it, ok = bisection_method(a, b, ftol=ftol, xtol=xtol, max_iter=100)
    print("Метод бисекции:")
    print(f"  Корень ~ {root:.12f} (iter={it}, ok={ok})")
    print(f"  Проверка f(root) ~ {f(root):.8e}\n")

    # 2) Fixed point: x_{n+1}=atan(x_n) (как в C++)
    root, it, ok = fixed_point_iteration_atan(x0=0.5, ftol=ftol, xtol=xtol, max_iter=200000)
    print("Метод простых итераций: x_{n+1} = atan(x_n)")
    print(f"  Корень ~ {root:.12f} (iter={it}, ok={ok})")
    print(f"  Проверка f(root) ~ {f(root):.8e}\n")

    # 3) Ньютон: для корня 0 кратность 3 -> используем модифицированный Ньютон
    root, it, ok = newton_method_modified(x0=0.5, ftol=ftol, xtol=xtol, max_iter=50, multiplicity=3)
    print("Метод Ньютона (модифицированный, m=3 для корня x=0):")
    print(f"  Корень ~ {root:.12f} (iter={it}, ok={ok})")
    print(f"  Проверка f(root) ~ {f(root):.8e}\n")

    # 4) Секущие
    root, it, ok = secant_method(x0=0.5, x1=-0.5, ftol=ftol, xtol=xtol, max_iter=100)
    print("Метод секущих:")
    print(f"  Корень ~ {root:.12f} (iter={it}, ok={ok})")
    print(f"  Проверка f(root) ~ {f(root):.8e}\n")

    print("=== Полином z^3 - 1 = 0 ===")
    print("Явные корни: 1, e^{i 2π/3}, e^{i 4π/3}.")
    print("Метод Ньютона для комплексных z.\n")

    initial_points = [
        2.0 + 0.0j,
        -1.0 + 1.0j,
        0.5 - 0.5j,
        -2.0 - 1.0j,
        1.0 + 0.5j,
    ]

    for z0 in initial_points:
        z_res, it, ok = newton_complex(z0, tol=1e-12, max_iter=100)
        idx = which_root(z_res)
        print(
            f"Старт z0 = ({z0.real:.8f},{z0.imag:.8f}) "
            f"=> итог z = ({z_res.real:.8f},{z_res.imag:.8f}), "
            f"близко к корню #{idx} (iter={it}, ok={ok})"
        )


if __name__ == "__main__":
    main()

