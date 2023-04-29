import copy


def array(data, copy_=True):
    import mynumpy as mynp

    if copy_:
        data = copy.deepcopy(data)
    return mynp.ndarray(data)
