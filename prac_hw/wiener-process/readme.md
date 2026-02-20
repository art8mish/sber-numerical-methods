
# Винеровский процесс

Требуется сгенерировать графики 1000 траекторий Винеровского процесса с $T = 1$, $\delta = 0.0001$.

## Решение

Для построение Винеровского процесса используется соотношение:
$$
B_{t+u} - B_t \sim N(0, u)
$$

Для шага $\delta$:
$$
B_{t_i + 1} = B_{t_i} + \sqrt{\delta} \cdot \xi_i,
$$
где 
- $B_{t_0} = 0$
- $\delta = 0.0001$ - шаг дискретизации
- $\xi_i$ - независимые случайные величины, распределенные по нормальному закону $\xi_i \sim N(0, u)$, т.е. $E\xi_i = 0$, $D\xi_i = 1$

Накопительная сумма:
$$
B_{t_k} = \sum_{i=1}^k \sqrt{\delta} \cdot \xi_i
$$

$B_0 = 0$ ($P$-п.н.)

## Запуск

```shell
python3 -m venv --prompt wiener .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 wiener.py
```

Получающийся график:

![Wiener_process](pic/wiener_1000_paths.png)