import pkg_resources
from symspellpy2.symspellpy import SymSpell
import pickle, re
import time


s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
    s = ''
    print(input_str.encode('utf-8'))
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def spell_correcting(dictionary_file, bigram_file, max__distance=5, prefix_length=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max__distance, prefix_length=prefix_length)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", dictionary_file)
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", bigram_file)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2, encoding='utf-8')
    return sym_spell


def load_engine(bigram, unigram, max__distance=5, prefix_length=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max__distance, prefix_length=prefix_length)

    sym_spell.load_dictionary_from_list(unigram, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary_from_list(bigram, term_index=0, count_index=2)
    return sym_spell


def load_address_correction(db_file, ):
    # with open(db_file, encoding='utf-8') as f:
    #     data = json.load(f)

    with open(db_file, 'rb') as handle:
        data = pickle.load(handle)
    unigram_province = data['unigram_province']
    bigram_province = data['bigram_province']
    province_correct = load_engine(bigram_province, unigram_province)
    full_db = {'province_correct': province_correct}
    full_db['provinces'] = {}

    for province in data['province']:
        provinces = {}
        name_province = province['name'].lower()
        unigram_district = province['unigram_district']
        bigram_district = province['bigram_district']
        district_correct = load_engine(bigram_district, unigram_district)
        # provinces[name_province] = {}
        provinces['district_correct'] = district_correct
        districts = {}
        for district in province['district']:
            name_district = district['name'].lower()


            unigram_ward = district['unigram_ward']
            bigram_ward = district['bigram_ward']
            unigram_streets = district['unigram_street']
            bigram_streets = district['bigram_street']
            unigram_all = district['unigram_all']
            bigram_all = district['bigram_all']
            if name_district == 'cửa lò':
                print(unigram_ward)
                print(bigram_ward)
            ward_correct = load_engine(bigram_ward, unigram_ward)
            streets_correct = load_engine(bigram_streets, unigram_streets)
            all_correct = load_engine(bigram_all, unigram_all)
            districts[name_district] = {}
            districts[name_district]["ward_correct"] = ward_correct
            districts[name_district]["streets_correct"] = streets_correct
            districts[name_district]["all_correct"] = all_correct
        provinces['districts'] = districts
        full_db['provinces'][name_province] = provinces
    print(full_db)

    with open('full_db' + '.pickle', 'wb') as handle:
        pickle.dump(full_db, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_address_correction(db_path):
    with open(db_path, 'rb') as handle:
        db = pickle.load(handle)
    return db

def correct_address(db, street=None, ward=None, district=None, city=None):

    def capitalize_(str_in):
        name_list = str_in.split(' ')
        name_out = []
        for word in name_list:
            name_out.append(word.capitalize())
        # print(name_out)
        # ouput.append()
        return ' '.join(name_out)
    output = {}
    city_suggestions = db['province_correct'].lookup_compound(city, max_edit_distance=5)
    # display suggestion term, edit distance, and term frequency
    for city_suggestion in city_suggestions:
        city_fixed = city_suggestion._term
        output['city'] = capitalize_(city_fixed)
        patterns_tp = ['thành phố', 'tp', 'tphố']
        for pattern in patterns_tp:
            group = re.search(pattern, city_fixed)
            if group is not None:
                city_fixed = city_fixed.replace(group.group(), '').lstrip()
        if city_fixed in db['provinces']:
            district_suggestions = db['provinces'][city_fixed]['district_correct'].lookup_compound(district,
                                                                                                   max_edit_distance=5)
            for district_suggestion in district_suggestions:
                district_fixed = district_suggestion._term
                output['district'] = capitalize_(district_fixed)
                patterns = ['quận', 'huyện', 'thành phố', 'tp', 'thị xã']
                for pattern in patterns:
                    group = re.search(pattern, district_fixed)
                    if group is not None:
                        district_fixed = district_fixed.replace(group.group(), '').lstrip()
                if district_fixed in db['provinces'][city_fixed]['districts']:
                    if ward != None:
                        ward_suggestions = db['provinces'][city_fixed]['districts'][district_fixed][
                            'ward_correct'].lookup_compound(remove_accents(ward),
                                                            max_edit_distance=5)
                        for ward_suggestion in ward_suggestions:
                            output['ward'] = capitalize_(ward_suggestion._term)
                    if street != None:
                        street_suggestions = db['provinces'][city_fixed]['districts'][district_fixed][
                            'streets_correct'].lookup_compound(remove_accents(street),
                                                               max_edit_distance=5)
                        for street_suggestion in street_suggestions:
                            output['street'] = capitalize_(street_suggestion._term)

    return output

if __name__ == "__main__":

    #with open('symspellpy/full_db.pickle', 'rb') as handle:
    # inp = 'Số 83 Thanh Lân, Phường hun T, Quận Hoang Mi, Hàn Nội'
    db = load_address_correction('full_db.pickle')
    begin = time.time()
    inp = 'Khối 1, Phường Thu Thuỷ, Thị Xã Cửu Lò, Nghệ An'

    out = correct_address(db=db,  ward='Thuu Thuỷ', district='Cửu Lò', city='Nghệ An')
    end = time.time()
    print('Time', end-begin)
    print(out)
