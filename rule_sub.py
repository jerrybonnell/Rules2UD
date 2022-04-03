
cat_info_list = []
cat_info_map = {}

def get_word_cat(unk):
    unk_value = ord(unk)
    print(f"unk: {unk} value: {unk_value}")
    # now we do search in the char.def file to determine which
    # category of unknown this char is from (which corresponds
    # to the cat_info_list and cat_info_map ds's)
    if unk_value in cat_info_map:
        return cat_info_map[unk_value]
    for entry in cat_info_list:
        if unk_value >= entry[0] and unk_value <= entry[1]:
            return entry[2]
    return 'DEFAULT'  # default is a mandatory category

def load_dictionary():
    with open('char.def', 'r') as f:
        cat_info_list = f.read().split('\n')
    cat_info_list = [l.split(' ') for l in cat_info_list][:-1]
    # convert the unicode representation to just an integer
    for line in cat_info_list:
        line[0] = int(line[0], 16)
        if len(line) > 2:
            line[1] = int(line[1], 16)
        else:
            # things strictly one value are added to the map
            cat_info_map[line[0]] = line[1]
    # delete all sublists that are strictly one value
    cat_info_list = [l for l in cat_info_list if len(l) == 3]
    # sort the list
    cat_info_list.sort(key=lambda x: x[0])
    return cat_info_list, cat_info_map

cat_info_list, cat_info_map = load_dictionary()
print(get_word_cat("^"))
