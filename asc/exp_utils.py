import ray


def exp_to_config(exp:ray.tune.Experiment):
    c = {}
    exp_config = exp.spec["config"]
    for k in exp_config:
        v = exp_config[k]
        if isinstance(v, dict) and "grid_search" in v:
            c[k] = v["grid_search"][0]
        else:
            c[k] = v

    return c