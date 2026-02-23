import math
import numpy as np


def analyze_precision(dtype, type_name: str):
    one = dtype(1.0)
    two = dtype(2.0)

    epsilon = one
    mantissa_bits = 0


    while dtype(one + dtype(epsilon / two)) != one:
        epsilon = dtype(epsilon / two)
        mantissa_bits += 1

    # Максимальная степень (пока не уйдём в inf)
    max_exp = 0
    val = one
    while not math.isinf(float(val)):
        val = dtype(val * two)
        max_exp += 1

    # Минимальная степень (пока не уйдём в 0, включая денормалы)
    min_exp = 0
    val = one
    while val != dtype(0.0):
        val = dtype(val / two)
        min_exp -= 1

    print(f"Тип: {type_name}")
    print(f"Машинное эпсилон (ε): {float(epsilon):.20g}")
    print(f"Число битов мантиссы: {mantissa_bits}")
    print(f"Минимальная степень: {min_exp}")
    print(f"Максимальная степень: {max_exp}")

    half_eps = dtype(epsilon / two)
    print("\nСравнение значений:")
    print(f"1              = {float(one):.20g}")
    print(f"1 + ε/2        = {float(dtype(one + half_eps)):.20g}")
    print(f"1 + ε          = {float(dtype(one + epsilon)):.20g}")
    print(f"1 + ε + ε/2    = {float(dtype(one + epsilon + half_eps)):.20g}")
    print("-----------------------------------\n")


def main():
    analyze_precision(np.float32, "float (numpy.float32)")
    analyze_precision(np.float64, "double (numpy.float64)")  # 64-bit float [web:6]

    # По желанию: Python float (обычно IEEE-754 double/binary64)
    analyze_precision(float, "double (python float)")


if __name__ == "__main__":
    main()

