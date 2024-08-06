# {'A':['a', 3], 'B':['b', 5]}  {'A':['a', 1], 'B':['c', 2], 'C':['c', 3]}
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
