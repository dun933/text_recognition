import pkg_resources
from symspellpy import SymSpell, Verbosity


def load_name_corection(dictionary_path, bigram_path):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # dictionary_path = pkg_resources.resource_filename(
    #     dictionary_path)
    # bigram_path = pkg_resources.resource_filename(
    #     bigram_path)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')
    return sym_spell


def correct_capital(raw, fixed):
    fixed = fixed.replace('_', ' ')
    if raw.islower():
        fixed = fixed.lower()
    elif raw.isupper():
        fixed = fixed.upper()
    else:
        fixed = fixed.split(' ')
        fixed = [word.capitalize() for word in fixed]
        fixed = ' '.join(fixed)
    return fixed
def correct_name(sym_spell, name_string=None):
    suggestions = sym_spell.lookup_compound(name_string, max_edit_distance=2)
    # display suggestion term, edit distance, and term frequency
    list_name = []
    for suggestion in suggestions:
        list_name.append(suggestion._term)
    output = []
    for name in list_name:
        if name is not None:
            name = correct_capital(raw=name_string,fixed=name)
        else:
            name = name_string
        output.append(name)
    return output

if __name__ == "__main__":
    input_term = "Nuyễn tành nm"
    sym_spell = load_name_corection("freq_name_dic.txt", "freq_name_bigram.txt")
    print(correct_name(sym_spell, input_term))
