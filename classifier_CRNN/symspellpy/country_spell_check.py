import pickle, os
import collections
from . import SymSpell, Verbosity

from . import correct_capital
# csv_data = 'countries_name/country-keyword-list.csv'


    
csv_data = 'countries_name/country-list.txt'

def load_country_correction(csv_data, edit_distance=5, prefix_length=7):
    with open(csv_data, 'rb') as f:
        data = f.readlines()
    data = [ctr.decode("utf-8").replace('\r\n','').lower().replace(' ','_') for ctr in data]
    c = collections.Counter()
    for i in data:
        c.update(set([i]))

    unigram = []
    for i in c:
        if i== 'vietnam':
            c[i] = 1000
        unigram.append([i, str(c[i] )])
    sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=prefix_length)
    sym_spell.load_dictionary_from_list(unigram, term_index=0, count_index=1)
    return  sym_spell

def correct_country(raw, sym_spell, edit_distance=5, verbosity=0):
    countr_fix_lookup = sym_spell.lookup(raw, max_edit_distance=edit_distance, verbosity=verbosity)
    # for ctr in countr_fix_lookup:
    #     print('lookup',ctr)
    if len(countr_fix_lookup)>0:
        output = correct_capital(raw=raw,fixed=countr_fix_lookup[0]._term)
    else:
        output = raw
    # countr_fix_lookup_compound = sym_spell.lookup_compound(raw,max_edit_distance=edit_distance)
    # for ctr in countr_fix_lookup_compound:
    #     print('compou', ctr)
    return output

if __name__ == "__main__":

    sym_spell = load_country_correction(csv_data=csv_data)
    correct_country('viit nm', sym_spell)
    # countr_fix = sym_spell.lookup('Vit Nam ',max_edit_distance=5,verbosity=0)
    # for ctr in countr_fix:
    #     print(ctr)
# print(countr_fix[0])

