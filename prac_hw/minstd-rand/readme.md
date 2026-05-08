# Оптимизация `minstd_rand`

Требуется реализовать свой `minstd_rand`, проверить совпадение с `std::minstd_rand` и сравнить последовательный и параллельный бенчмарк Монте-Карло для числа `pi` на `100000000` двумерных векторов.

## Решение

Линейный конгруэнтный генератор `minstd_rand`:

```math
x_{i+1} = 48271 * x_i mod (2^31 - 1)
```

## Сборка и запуск

```shell
cd prac_hw/minstd-rand
make
./minstd_rand
```

Параметры сборки: `-O3 -Wall -Wextra -std=c++17 -march=native -fopenmp`.

## Результаты

### Генератор

```shell
Verification with std::minstd_rand: PASS
Benchmark vectors: 100000000
OpenMP threads: 12

Generator clocks/element (N=2^24)
scalar_cpe=7.060878 (cycles=118461882)
vector_cpe=4.370676 (cycles=73327782)
Speedup (scalar/vector generator): 1.615512
```

Последовательность полностью совпадает с `std::minstd_rand`. Для векторизованного через AVX генератора `fp32` коэффициент ускорения `scalar/vector`: `1.607552` (`7.06 -> 4.37` clocks/element).


### Бенчмарк

```shell
Monte Carlo benchmark results:
Case      | hits      | pi       | cycles
-----------------------------------------------
scalar    | 78537175 | 3.141487 | 1124442792
parallel  | 78537175 | 3.141487 | 133762596

Speedup (scalar/parallel): 8.406257
Abs(pi_scalar - pi_parallel): 0.000000
Hits equal: true
Vectorization note: AVX2 is enabled for fp32 conversion/store path.
```


Параллельная версия (OpenMP 12 threads) Monte-Carlo корректна (`hits` совпадают, разность по `pi` равна нулю). Коэффициент ускорения **8.4**



