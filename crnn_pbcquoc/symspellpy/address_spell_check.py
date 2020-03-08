import pkg_resources
from . import SymSpell, Verbosity
from collections import defaultdict
import os, json, pickle, re
import pandas
import collections

from . import correct_capital


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


def load_engine(bigram=None, unigram=None, max__distance=5, prefix_length=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max__distance, prefix_length=prefix_length)
    sym_spell.load_dictionary_from_list(unigram, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary_from_list(bigram, term_index=0, count_index=2)
    return sym_spell

def extractDigits(lst): 
    return list(map(lambda el:[el], lst)) 

def export_freq_dic(words_prov):
    c = collections.Counter()
    words = list(words_prov)
    for i in words:
        c.update(set(i))
    unigram_list = []
    for i in c:
        # if i != '' and i != ' ':
        unigram_list.append([i, 1])
    return unigram_list

def export_freq_bigram(list_prov):
    c = collections.Counter()
    for i in list_prov:
        c.update(set(zip(i[:-1], i[1:])))
    bigram_list = []
    for i in c:
        list_i = list(i)
        list_i.append(c[i])
        bigram_list.append(list_i)
    return bigram_list

def load_big_engine(corpus, max__distance=5, prefix_length=7):
    sym_spell = SymSpell(max_dictionary_edit_distance=max__distance, prefix_length=prefix_length)

    # print(corpus)
    corpus = extractDigits(corpus)
    unigram = export_freq_dic(corpus)
    bigram = export_freq_bigram(corpus)
    # print(unigram)
    # print(bigram)
    sym_spell.load_dictionary_from_list(unigram, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary_from_list(bigram, term_index=0, count_index=2)
    return sym_spell

"""def load_address_correction(db_file, db_out):
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
        provinces['district_correct'] = district_correct
        districts = {}
        for district in province['district']:
            name_district = district['name'].lower()


            unigram_ward = district['unigram_ward']
            bigram_ward = district['bigram_ward']
            unigram_streets = district['unigram_street']
            bigram_streets = district['bigram_street']
            # unigram_all = district['unigram_all']
            # bigram_all = district['bigram_all']
            ward_correct = load_engine(bigram_ward, unigram_ward)
            streets_correct = load_engine(bigram_streets, unigram_streets)
            # all_correct = load_engine(bigram_all, unigram_all)
            districts[name_district] = {}
            districts[name_district]["ward_correct"] = ward_correct
            districts[name_district]["streets_correct"] = streets_correct
            # districts[name_district]["all_correct"] = all_correct
        provinces['districts'] = districts
        full_db['provinces'][name_province] = provinces
    # print(full_db)
    # suggestions = province_correct.lookup_compound('Hà Giaiagg', max_edit_distance=5)
    # # display suggestion term, edit distance, and term frequency
    # for suggestion in suggestions:
    #     print(suggestion)

    # with open(db_out + '.json', 'w', encoding='utf-8') as f:
    #     json.dump(full_db, f, ensure_ascii=False, indent=2)
    with open(db_out, 'wb') as handle:
        pickle.dump(full_db, handle, protocol=pickle.HIGHEST_PROTOCOL)



def correct_address(db, street=None, ward=None, district=None, city=None):
    output = {}
    city_suggestions = db['province_correct'].lookup_compound(city, max_edit_distance=5)
    # display suggestion term, edit distance, and term frequency
    for city_suggestion in city_suggestions:
        # print(city_suggestion._term)
        city_fixed = city_suggestion._term
        output['city'] = city_fixed
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
                output['district'] = district_fixed
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
                            # outp.append(ward_suggestion._term)
                            output['ward'] = ward_suggestion._term
                    if street != None:
                        street_suggestions = db['provinces'][city_fixed]['districts'][district_fixed][
                            'streets_correct'].lookup_compound(remove_accents(street),
                                                               max_edit_distance=5)
                        for street_suggestion in street_suggestions:
                            # print(street_suggestion._term)
                            # outp.append(street_suggestion._term)
                            output['street'] = street_suggestion._term

    return output"""

def load_address_correction(db_file, db_out):
    # with open(db_file, encoding='utf-8') as f:
    #     data = json.load(f)

    with open(db_file, 'rb') as handle:
        data = pickle.load(handle)
    unigram_province = data['unigram_province']
    bigram_province = data['bigram_province']
    province_correct = load_engine(bigram_province, unigram_province)
    full_db = {'province_correct': province_correct}
    full_db['provinces'] = {}



    unigram_district = data['unigram_district']
    bigram_district = data['bigram_district']
    district_correct = load_engine(bigram_district, unigram_district)
    # full_db = {'district_correct': district_correct}
    full_db['district_correct'] = district_correct


    unigram_ward = data['unigram_ward']
    bigram_ward = data['bigram_ward']
    ward_correct = load_engine(bigram_ward, unigram_ward)
    # full_db = {'ward_correct': ward_correct}
    full_db['ward_correct'] = ward_correct

    unigram_streets = data['unigram_street']
    bigram_streets = data['bigram_street']
    streets_correct = load_engine(bigram_streets, unigram_streets)
    # full_db = {'streets_correct': streets_correct}
    full_db['streets_correct'] = streets_correct

    for province in data['province']:
        provinces = {}
        name_province = province['name'].lower()
        unigram_district = province['unigram_district']
        bigram_district = province['bigram_district']
        district_correct = load_engine(bigram_district, unigram_district)
        provinces['district_correct'] = district_correct
        districts = {}
        for district in province['district']:
            name_district = district['name'].lower()


            unigram_ward = district['unigram_ward']
            bigram_ward = district['bigram_ward']
            unigram_streets = district['unigram_street']
            bigram_streets = district['bigram_street']
            # unigram_all = district['unigram_all']
            # bigram_all = district['bigram_all']
            ward_correct = load_engine(bigram_ward, unigram_ward)
            streets_correct = load_engine(bigram_streets, unigram_streets)
            # all_correct = load_engine(bigram_all, unigram_all)
            districts[name_district] = {}
            districts[name_district]["ward_correct"] = ward_correct
            districts[name_district]["streets_correct"] = streets_correct
            districts[name_district]["corpus_street"] = district['corpus_street']
            # districts[name_district]["all_correct"] = all_correct
        provinces['districts'] = districts
        full_db['provinces'][name_province] = provinces
    unigram_district = data['unigram_district']
    bigram_district = data['bigram_district']
    district_correct = load_engine(bigram_district, unigram_district)
    # full_db = {'district_correct': district_correct}
    full_db['district_correct'] = district_correct


    unigram_ward = data['unigram_ward']
    bigram_ward = data['bigram_ward']
    ward_correct = load_engine(bigram_ward, unigram_ward)
    # full_db = {'ward_correct': ward_correct}
    full_db['ward_correct'] = ward_correct

    unigram_streets = data['unigram_street']
    bigram_streets = data['bigram_street']
    streets_correct = load_engine(bigram_streets, unigram_streets)
    # full_db = {'streets_correct': streets_correct}
    full_db['streets_correct'] = streets_correct
    # print(full_db)
    # suggestions = province_correct.lookup_compound('Hà Giaiagg', max_edit_distance=5)
    # # display suggestion term, edit distance, and term frequency
    # for suggestion in suggestions:
    #     print(suggestion)

    # with open(db_out + '.json', 'w', encoding='utf-8') as f:
    #     json.dump(full_db, f, ensure_ascii=False, indent=2)
    with open(db_out, 'wb') as handle:
        pickle.dump(full_db, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # oringin 
# def correct_address(db, street=None, ward=None, district=None, city=None):
#     output = {}
#     city_suggestions = db['province_correct'].lookup_compound(city, max_edit_distance=5)
#     # display suggestion term, edit distance, and term frequency
#     for city_suggestion in city_suggestions:
#         # print(city_suggestion._term)
#         city_fixed = city_suggestion._term
#         output['city'] = city_fixed
#         patterns_tp = ['thành phố', 'tp', 'tphố']
#         for pattern in patterns_tp:
#             group = re.search(pattern, city_fixed)
#             if group is not None:
#                 city_fixed = city_fixed.replace(group.group(), '').lstrip()
#         if city_fixed in db['provinces']:
#             district_suggestions = db['provinces'][city_fixed]['district_correct'].lookup_compound(district,
#                                                                                                    max_edit_distance=5)
#             for district_suggestion in district_suggestions:
#                 district_fixed = district_suggestion._term
#                 output['district'] = district_fixed
#                 patterns = ['quận', 'huyện', 'thành phố', 'tp', 'thị xã']
#                 for pattern in patterns:
#                     group = re.search(pattern, district_fixed)
#                     if group is not None:
#                         district_fixed = district_fixed.replace(group.group(), '').lstrip()
#                 if district_fixed in db['provinces'][city_fixed]['districts']:
#                     if ward != None:
#                         ward_suggestions = db['provinces'][city_fixed]['districts'][district_fixed][
#                             'ward_correct'].lookup_compound(remove_accents(ward),
#                                                             max_edit_distance=5)
#                         for ward_suggestion in ward_suggestions:
#                             # outp.append(ward_suggestion._term)
#                             output['ward'] = ward_suggestion._term
#                     if street != None:
#                         street_suggestions = db['provinces'][city_fixed]['districts'][district_fixed][
#                             'streets_correct'].lookup_compound(remove_accents(street),
#                                                                max_edit_distance=5)
#                         for street_suggestion in street_suggestions:
#                             # print(street_suggestion._term)
#                             # outp.append(street_suggestion._term)
#                             output['street'] = street_suggestion._term

#     return output

def correct_address(db, csv_file, street=None, ward=None, district=None, city=None, max_edit_distance=5):
    def city_non_pre(city_fixed):
        city_fixed = city_fixed.lower()
        patterns_tp = ['thành phố', 'tp', 'tphố', 'tỉnh']
        for pattern in patterns_tp:
            group = re.search(pattern, city_fixed)
            if group is not None:
                city_fixed = city_fixed.replace(group.group(), '').lstrip()
        return city_fixed.replace(' ', '_')
    def district_non_pre(district_fixed):
        district_fixed = district_fixed.lower()
        patterns_tp = ['quận', 'huyện', 'tphố', 'thành phố', 'tp', 'tt', 'thị trấn', 'thị xã', 'tx']
        for pattern in patterns_tp:
            group = re.search(pattern, district_fixed)
            if group is not None:
                district_fixed = district_fixed.replace(group.group(), '').lstrip()
        return district_fixed.replace(' ', '_')
    def ward_non_pre(ward_fixed):
        ward_fixed = ward_fixed.lower()
        patterns_tp = ['phường', 'xã', 'phố', 'thị trấn', 'tt', 'thị xã', 'tx']
        for pattern in patterns_tp:
            group = re.search(pattern, ward_fixed)
            if group is not None:
                ward_fixed = ward_fixed.replace(group.group(), '').lstrip()
        return ward_fixed.replace(' ', '_')
    def street_non_pre(street_fixed):
        street_fixed = street_fixed.lower()
        patterns_tp = ['số', 'đường', 'nhà', 'phố', 'street', 'thôn']
        for pattern in patterns_tp:
            group = re.search(pattern, street_fixed)
            if group is not None:
                print(group.group())
                street_fixed = street_fixed.replace(group.group(), '').lstrip()
        return street_fixed.replace(' ', '_')

    output = {}
    output['city_fixed'] = None
    output['district_fixed'] = None
    output['ward_fixed'] = None
    output['street_fixed'] = None

    data_df = pandas.read_csv(csv_file)
    data_df = data_df.drop('Mã TP', 1)
    data_df = data_df.drop('Mã QH', 1)
    data_df = data_df.drop('Mã PX', 1)
    data_df = data_df.drop('Cấp', 1)
    data_df = data_df.astype(str).apply(lambda x: x.str.lower())
    data_df = data_df.astype(str).apply(lambda x: x.str.replace(' ', '_'))
    # print(data_df)
    if city:
        city_tk = city_non_pre(city)
        if city_tk in data_df['Tỉnh Thành Phố'].values:
            output['city_fixed'] = city_tk
        else:
            city_suggestions = db['province_correct'].lookup_compound(city_tk, max_edit_distance=max_edit_distance)
            for city_suggestion in city_suggestions:
                city_fixed = city_suggestion._term
                print(city_fixed)
                # if '_' in city_fixed:
                #     city_fixed = ' '.join(list(city_fixed.split('_')))
                if city_fixed in data_df['Tỉnh Thành Phố'].values:
                    output['city_fixed'] = city_fixed

    if district:
        district_tk = district_non_pre(district)
        # print(district_tk)
        if output['city_fixed']:
            df_city = data_df.loc[data_df['Tỉnh Thành Phố'].isin([output['city_fixed']])]
            
            if district_tk in df_city['Quận Huyện'].values:
                output['district_fixed'] = district_tk

            else:
                city_fixed_t = output['city_fixed'].replace(' ','_')
                district_suggestions = db['provinces'][city_fixed_t]['district_correct'].lookup_compound(district_tk,
                                                                                                   max_edit_distance=max_edit_distance)
                for district_suggestion in district_suggestions:
                    district_fixed = district_suggestion._term
                    # if '_' in district_fixed:
                    #     district_fixed = ' '.join(list(district_fixed.split('_')))
                    if district_fixed in df_city['Quận Huyện'].values:
                        output['district_fixed'] = district_fixed
        else:
            if district_tk in data_df['Quận Huyện'].values:
                df_district = data_df.loc[data_df['Quận Huyện'].isin([district_tk])]
                list_city = df_district['Tỉnh Thành Phố'].unique()
                output['city_fixed'] = list_city[0]
                output['district_fixed'] = district_tk
            else:
                district_suggestions = db['district_correct'].lookup_compound(district_tk,max_edit_distance=max_edit_distance)
                for district_suggestion in district_suggestions:
                    district_fixed = district_suggestion._term
                    print(district_fixed)
                    # if '_' in district_fixed:
                    #     district_fixed = ' '.join(list(district_fixed.split('_')))
                    if district_fixed in data_df['Quận Huyện'].values:
                        output['district_fixed'] = district_fixed
                        df_district = data_df.loc[data_df['Quận Huyện'].isin([district_fixed])]
                        list_city = df_district['Tỉnh Thành Phố'].unique()
                        output['city_fixed'] = list_city[0]
    if ward:
        ward_tk = ward_non_pre(ward)
        if output['city_fixed'] and output['district_fixed']:
            df_city = data_df.loc[data_df['Tỉnh Thành Phố'].isin([output['city_fixed']])]
            df_district = df_city.loc[df_city['Quận Huyện'].isin([output['district_fixed']])]
            # print(df_district)
            if ward_tk in df_district['Phường Xã'].values:
                output['ward_fixed'] = ward_tk
            else:
                city_fixed_t = output['city_fixed'].replace(' ','_')
                district_fixed_t = output['district_fixed'].replace(' ','_')
                # print(ward_tk)
                ward_suggestions = db['provinces'][city_fixed_t]['districts'][district_fixed_t][
                            'ward_correct'].lookup_compound(ward_tk,
                                                            max_edit_distance=max_edit_distance)
                for ward_suggestion in ward_suggestions:
                    ward_fixed = ward_suggestion._term
                    # print(ward_fixed)
                    if ward_fixed in df_district['Phường Xã'].values:
                        output['ward_fixed'] = ward_fixed
                if output['ward_fixed'] == None:
                    ward_suggestions_ = db['provinces'][city_fixed_t]['districts'][district_fixed_t][
                                'ward_correct'].lookup(ward_tk,verbosity=0,
                                                                max_edit_distance=max_edit_distance)
                    for ward_suggestion in ward_suggestions_:
                        ward_fixed = ward_suggestion._term
                        # print(ward_fixed)
                        if ward_fixed in df_district['Phường Xã'].values:
                            output['ward_fixed'] = ward_fixed
        elif output['city_fixed'] and output['district_fixed']==None:
            df_city = data_df.loc[data_df['Tỉnh Thành Phố'].isin([output['city_fixed']])]
            
            if ward in df_city['Phường Xã'].values:
                output['ward_fixed'] = ward
            else:
                city_fixed_t = output['city_fixed'].replace(' ','_')
                # district_fixed_t = output['district_fixed'].replace(' ','_')
                # print(ward)
                list_ward = df_city['Phường Xã'].unique()
                # print(list_ward)
                ward_correct = load_big_engine(list_ward)
                # ward_suggestions = db['provinces'][city_fixed_t]['districts'][district_fixed_t][
                #             'ward_correct'].lookup_compound(ward, max_edit_distance=max_edit_distance)
                ward_suggestions = ward_correct.lookup_compound(ward,max_edit_distance=max_edit_distance)
                for ward_suggestion in ward_suggestions:
                    ward_fixed = ward_suggestion._term
                    # print(ward_fixed)
                    # if '_' in ward_fixed:
                    #     ward_fixed = ' '.join(list(ward_fixed.split('_')))
                    if ward_fixed in df_city['Phường Xã'].values:
                        output['ward_fixed'] = ward_fixed
                        df_district = df_city.loc[df_city['Phường Xã'].isin([ward_fixed])]
                        list_district = df_district['Quận Huyện'].unique()
                        output['district_fixed'] = list_district[0]
        elif output['city_fixed']==None and output['district_fixed']:
            df_district = data_df.loc[data_df['Quận Huyện'].isin([output['district_fixed']])]
            # print(df_district)
            if ward in df_district['Phường Xã'].values:
                output['ward_fixed'] = ward
                df_district = df_district.loc[df_district['Phường Xã'].isin([ward])]
                list_district = df_district['Quận Huyện'].unique()
                output['district_fixed'] = list_district[0]
    if street:
        street_tk = street_non_pre(street)
        # print(street_tk)
        if output['city_fixed'] and output['district_fixed']:
            city_fixed_t = output['city_fixed'].replace(' ','_')
            district_fixed_t = output['district_fixed'].replace(' ','_')
            corpus_street = db['provinces'][city_fixed_t]['districts'][district_fixed_t]['corpus_street']
            # print(corpus_street)
            if street_tk in corpus_street:
                output['street_fixed'] = street_tk
            else:
                # print(street_tk)
                street_suggestions = db['provinces'][city_fixed_t]['districts'][district_fixed_t][
                            'streets_correct'].lookup(street_tk,max_edit_distance=max_edit_distance,verbosity=0)
                # for street_suggestion in street_suggestions:

                # print(street_suggestion)
                if len(street_suggestions)>0:
                    street_fixed = street_suggestions[0]._term
                    # if '_' in street_fixed:
                    #     street_fixed = ' '.join(list(street_fixed.split('_')))
                    if street_fixed in corpus_street:
                        output['street_fixed'] = street_fixed
    

    if output['city_fixed'] is not None:
        output['city_fixed'] = correct_capital(raw=city,fixed=output['city_fixed'])
    else:
        output['city_fixed'] = city

    if output['district_fixed'] is not None:
        output['district_fixed'] = correct_capital(raw=district,fixed=output['district_fixed'])
    else:
        output['district_fixed'] = district

    if output['ward_fixed'] is not None:
        output['ward_fixed'] = correct_capital(raw=ward,fixed=output['ward_fixed'])
    else:
        output['ward_fixed'] = ward

    if output['street_fixed'] is not None:
        output['street_fixed'] = correct_capital(raw=street,fixed=output['street_fixed'])
    else:
        output['street_fixed'] = street

    return output

# def load_address_correction(db_file, db_out):
#     # with open(db_file, encoding='utf-8') as f:
#     #     data = json.load(f)
#     full_db = {}
#     with open(db_file, 'rb') as handle:
#         data = pickle.load(handle)
#     unigram_province = data['unigram_province']
#     bigram_province = data['bigram_province']
#     province_correct = load_engine(bigram_province, unigram_province)
#     # full_db = {'province_correct': province_correct}
#     full_db['province_correct'] = province_correct

#     unigram_district = data['unigram_district']
#     bigram_district = data['bigram_district']
#     district_correct = load_engine(bigram_district, unigram_district)
#     # full_db = {'district_correct': district_correct}
#     full_db['district_correct'] = district_correct


#     unigram_ward = data['unigram_ward']
#     bigram_ward = data['bigram_ward']
#     ward_correct = load_engine(bigram_ward, unigram_ward)
#     # full_db = {'ward_correct': ward_correct}
#     full_db['ward_correct'] = ward_correct

#     unigram_streets = data['unigram_street']
#     bigram_streets = data['bigram_street']
#     streets_correct = load_engine(bigram_streets, unigram_streets)
#     # full_db = {'streets_correct': streets_correct}
#     full_db['streets_correct'] = streets_correct

#     # with open(db_out + '.json', 'w', encoding='utf-8') as f:
#     #     json.dump(full_db, f, ensure_ascii=False, indent=2)
#     with open(db_out, 'wb') as handle:
#         pickle.dump(full_db, handle, protocol=pickle.HIGHEST_PROTOCOL)



# def correct_address(db, street=None, ward=None, district=None, city=None):
#     output = {}
#     if city != None:
#         city_suggestions = db['province_correct'].lookup_compound(city, max_edit_distance=5)
#         # display suggestion term, edit distance, and term frequency
#         for city_suggestion in city_suggestions:
#             city_fixed = city_suggestion._term
#             output['city'] = city_fixed
#     if district != None:
#         district_suggestions = db['district_correct'].lookup_compound(district,max_edit_distance=5)
#         for district_suggestion in district_suggestions:
#             district_fixed = district_suggestion._term
#             output['district'] = district_fixed
#     if ward != None:
#         ward_suggestions = db['ward_correct'].lookup_compound(ward,max_edit_distance=5)
#         for ward_suggestion in ward_suggestions:
#             output['ward'] = ward_suggestion._term
#     if street != None:
#         street_suggestions = db['streets_correct'].lookup_compound(street,max_edit_distance=5)
#         for street_suggestion in street_suggestions:
#             output['street'] = street_suggestion._term

#     return output

if __name__ == "__main__":
    # string = "h1DF346 123FE453 3f3g6hj7j5v3 hasdf@asdf r3 r@ 555555 @ hello onlyletters"
    # rx = re.compile(r'(?=\d+[A-Za-z]+[\w]+|[a-zA-Z]+[\w]+)[\w]{2,}')
    # print(rx.findall(string))
    load_address_correction('../vietnamese_placedb/database.pickle')

    with open('full_db' + '.pickle', 'rb') as handle:
        db = pickle.load(handle)
    # inp = 'Số 83 Thanh Lân, Phường hun T, Quận Hoang Mi, Hàn Nội'
    inp = 'Khối 1, Phường Thu Thuỷ, Thị Xã Cửu Lò, Nghệ An'
    out = correct_address(db=db, city='Nghệ An', district='Thị Xã Cửu Lò', ward='Phường Thu Thuỷ')
    print(out)
