from symspellpy2.symspellpy import SymSpell


def load_name_corection(dictionary_path, bigram_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # dictionary_path = pkg_resources.resource_filename(
    #     dictionary_path)
    # bigram_path = pkg_resources.resource_filename(
    #     bigram_path)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')
    return sym_spell

def correct_name(sym_spell, input_str=None):
    suggestions = sym_spell.lookup_compound(input_str, max_edit_distance=2)
    # display suggestion term, edit distance, and term frequency
    ouput = []
    for suggestion in suggestions:
        name = suggestion._term
        name_list = name.split(' ')
        name_out = []
        for word in name_list:
            name_out.append(word.capitalize())
        # print(name_out)
        ouput.append(' '.join(name_out))
    return ouput

if __name__ == "__main__":
    input_str = "Hoàng Đình THơi"
    sym_spell = load_name_corection("freq_name_dic.txt", "freq_name_bigram.txt")
    print(correct_name(sym_spell, input_str))
