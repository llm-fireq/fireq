
def find_by_index(pair_list, index):
    ret_list = []
    for pair in pair_list:
        if pair.exist_not_mapped: continue
        if pair.index == index:
            ret_list.append(pair)
    return ret_list

def exist_endswith_in_list(tlist, name):
    for tname in tlist:
        if tname.endswith(name):
            return True
    return False

def find_by_from_name(pair_list, from_name):
    ret_list = []
    for pair in pair_list:
        if pair.exist_not_mapped: continue

        if exist_endswith_in_list(pair._from, from_name):
            ret_list.append(pair)
    return ret_list
        
def find_by_to_name(pair_list, to_name):
    ret_list = []
    for pair in pair_list:
        if pair.exist_not_mapped: continue

        if exist_endswith_in_list(pair._to, to_name):
            ret_list.append(pair)
    return ret_list

def find_by_from_or_to_name(pair_list, name):
    name_from = find_by_from_name(pair_list, name)
    name_to = find_by_to_name(pair_list, name)

    return list(set(name_from).union(set(name_to)))

def find_by_adjacent_names(pair_list, name1, name2):
    name1_set = set(find_by_from_or_to_name(pair_list, name1))
    name2_set = set(find_by_from_or_to_name(pair_list, name2))

    return list(name1_set.union(name2_set))

def find_by_allowed_calib(pair_list, allowed_calib):
    assert type(allowed_calib) in [str, list, tuple]
    if type(allowed_calib) == str:
        assert allowed_calib in ["rotated", "reordered", "smoothed"]
        allowed_calib = [allowed_calib]
    else:
        for calib in allowed_calib:
            assert calib in ["rotated", "reordered", "smoothed"]
    ret_list = []
    for pair in pair_list:
        allowed = True
        if pair.not_allowed is None:
            ret_list.append(pair)
            continue
        for calib in allowed_calib:
            if calib in pair.not_allowed:
                allowed = False
                
        if allowed:
            ret_list.append(pair)

    return ret_list
        