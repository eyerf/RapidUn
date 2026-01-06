import time

from RapidIn.calc_inner import s_test

IGNORE_INDEX = -100


def calc_s_test_single(
    model,
    z_test,
    t_test,
    input_len,
    train_loader,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    need_reshape=True,
):
    min_nan_depth = recursion_depth

    res, nan_depth = s_test(
        z_test,
        t_test,
        input_len,
        model,
        train_loader,
        gpu=gpu,
        damp=damp,
        scale=scale,
        recursion_depth=recursion_depth,
        need_reshape=need_reshape,
    )
    min_nan_depth = min(min_nan_depth, nan_depth)

    for _ in range(1, r):
        cur, nan_depth = s_test(
            z_test,
            t_test,
            input_len,
            model,
            train_loader,
            gpu=gpu,
            damp=damp,
            scale=scale,
            recursion_depth=recursion_depth,
            need_reshape=need_reshape,
        )
        res = res + cur
        min_nan_depth = min(min_nan_depth, nan_depth)

    if min_nan_depth != recursion_depth:
        print(
            f"Warning: NaN encountered at depth {min_nan_depth}; "
            f"current recursion_depth = {recursion_depth}"
        )

    res = res / r
    return res