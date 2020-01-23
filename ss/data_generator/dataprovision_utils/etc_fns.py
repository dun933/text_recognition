def prepare_list_from_label_file(label_filepath):
    print('etc_fns.prepare_list_from_label_file:',label_filepath)
    with open(label_filepath, encoding="utf8") as fd:
        rdata = fd.readline()
    if rdata[-1]=='\n':
        rdata = rdata[0:-1]
    tolist = list(str(rdata))
    return tolist

def fetch_word_list_from_txt_file(txt_filepath, mode=1):
    print('etc_fns.fetch_word_list_from_txt_file:',txt_filepath,', mode:',mode)
    with open(txt_filepath,'r', encoding='utf-8') as fd:
        all_lines = fd.readlines()

    word_list=[]
    for line in all_lines:
        if line is None:
            continue
        if line[-1] == "\n":
            line = line[:-1]

        if line[-1] == " ":
            line = line[:-1]

        if not line:
            continue
        if (mode == 1):
            splits = line.split(' ')
            word_list.extend(splits)
        if (mode == 2):
            word_list.append(line)
    # for w in word_list:
    #     if '\n' in w:
    #         print(w)

    return word_list
