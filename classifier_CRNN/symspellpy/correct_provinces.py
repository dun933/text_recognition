import pkg_resources
from symspellpy import SymSpell, Verbosity

import wordsegUtil

sym_spell = SymSpell(max_dictionary_edit_distance=5, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "freq_provinces_dic.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "freq_provinces_bigram.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
# sym_spell.load_dictionary('C:/Users/nt.anh6/PycharmProjects/aicr_vn/nlp_model/sp/ell_checker/dict/vi_full.txt', term_index=0, count_index=1, encoding='utf-8')
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')

# lookup suggestions for multi-word input strings (supports compound
# splitting & merging)
input_term = "34/1 ấp Hộp Lên, Xã Bò Đầnc, Huyện Hói Nêân, TP Hồ Cihí Minh"

# input_term = wordsegUtil.cleanLine(input_term)
# print(input_term)
# # # max edit distance per lookup (per single word, not per whole input string)
# # # suggestions = sym_spell.lookup(input_term, Verbosity.TOP, max_edit_distance=5, include_unknown=False)
# # suggestions = sym_spell.lookup_compound(input_term, max_edit_distance=5)
# # # display suggestion term, edit distance, and term frequency
# # for suggestion in suggestions:
# #     print(suggestion)

# from difflib import SequenceMatcher
#
#
# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()


# print(similar("nội", 'nội'))
import re


# result = re.sub('[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', u'', 'nội')
# print(result)
# print(similar("nội", result))

def spell_correcting(dictionary_file, bigram_file, max__distance=5, prefix_length=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max__distance, prefix_length=prefix_length)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", dictionary_file)
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", bigram_file)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')
    return sym_spell

prov_spell = spell_correcting(dictionary_file="freq_provinces_dic.txt", bigram_file="freq_provinces_bigram.txt", max__distance=5, prefix_length=7)

suggestions = prov_spell.lookup_compound(input_term, max_edit_distance=5)
# display suggestion term, edit distance, and term frequency
for suggestion in suggestions:
    print(suggestion)