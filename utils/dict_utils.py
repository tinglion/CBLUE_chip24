# 同名合并，取最大的，相同相加
#  {'A':['a', 3], 'B':['b', 5]}  {'A':['a', 1], 'B':['c', 2], 'C':['c', 3]}
# => {'A':['a', 4], 'B':['b', 5], 'C':['c', 3]}
def merge_dict_2(d1, d2):
    if not d1:
        return d2
    if not d2:
        return d1

    result = {}
    for key in d1:
        if key in d2:
            if d2[key][0] == d1[key][0]:
                result[key] = [d1[key][0], d2[key][1] + d1[key][1]]
            else:
                if d2[key][1] > d1[key][1]:
                    result[key] = d2[key]
                else:
                    result[key] = d1[key]
        else:
            result[key] = d1[key]
    return result


# 同名合并，取最大的，相同相加
#  {'A':{'a': 3}, 'B':{'b', 5}}  {'A':{'a': 1}, 'B':{'c': 2}, 'C':{'c': 3}]}
# => {'A':{'a':4}, 'B':{'b': 5, 'c':2}, 'C':{'c': 3}}
def merge_dict_3(d1, d2, min_value=0):
    if not d1:
        d1 = {}
    if not d2:
        return d1

    for key in d2:
        if not key in d1:
            d1[key] = d2[key]
        else:
            for k2 in d2[key]:
                if d2[key][k2] >= min_value:
                    if k2 in d1[key]:
                        d1[key][k2] += d2[key][k2]
                    else:
                        d1[key][k2] = d2[key][k2]
    return d1


def revise_dict(response_obj, cmap, cmap_r):
    response_obj2 = {}
    for k1 in response_obj:
        if len(response_obj[k1]) < 2:
            continue

        k2 = response_obj[k1][0]
        # qianfan format bug
        if isinstance(k2, list):
            k2 = k2[0]

        if k2 not in cmap and k2 in cmap_r:
            k2 = cmap_r[k2]

        response_obj2[k1] = [k2, response_obj[k1][1]]
    return response_obj2


def revise_dict3(response_obj, cmap, cmap_r):
    response_obj2 = {}
    for k1 in response_obj:
        for k2 in response_obj[k1]:
            if response_obj[k1][k2] > 0:
                if k1 not in response_obj2:
                    response_obj2[k1] = {}

                real_k2 = k2
                # qianfan format bug
                if isinstance(k2, list):
                    real_k2 = k2[0]
                if real_k2 not in cmap and real_k2 in cmap_r:
                    real_k2 = cmap_r[real_k2]

                response_obj2[k1][real_k2] = response_obj[k1][k2]
    return response_obj2
