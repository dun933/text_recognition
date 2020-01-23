
class Charset():
    def __init__(self, **kwargs):
        print('Charset.Init')
        vietnam_file_path = kwargs["vietnam_file_path"]
        symbol_file_path = kwargs["symbol_file_path"]
        bg_symbol_file_path = kwargs["bg_symbol_file_path"]

        if symbol_file_path is not None:
            with open(symbol_file_path) as fp:
                self.symbol_unicode_list = [ord(c) for c in fp.read(-1)]
            self.symbol_unicode_set = {c for c in self.symbol_unicode_list}

        else:
            symbols = '*:,@.-(#%\'")/~!^&_+={}[]|\;<>?※₩$¢¥€£₫'
            self.symbol_unicode_set = {ord(s) for s in symbols}
            self.symbol_unicode_list = [ord(s) for s in symbols]

        if bg_symbol_file_path is not None:
            with open(bg_symbol_file_path) as fp:
                self.bg_symbol_unicode_list = [ord(c) for c in fp.read(-1)]
            self.bg_symbol_unicode_set = {c for c in self.bg_symbol_unicode_list}

        else:
            bg_symbols = "| "
            # bg_symbols = "|➀➁➂➃➄➅➆➇➈➉①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
            self.bg_symbol_unicode_set = {ord(s) for s in bg_symbols}
            self.bg_symbol_unicode_list = [ord(s) for s in bg_symbols]

        self.alphabet_unicode_set = set(range(ord('a'),ord('z')+1)) | set(range(ord('A'),ord('Z')+1))
        self.alphabet_unicode_list = list(range(ord('a'),ord('z')+1)) + list(range(ord('A'),ord('Z')+1))

        self.vietnam_unicode_set = set()
        self.vietnam_unicode_list = list()
        if vietnam_file_path is not None:
            with open(vietnam_file_path, encoding="utf8") as fp:
                self.vietnam_unicode_list = [ord(c) for c in fp.read(-1)]
            self.vietnam_unicode_set = {c for c in self.vietnam_unicode_list}

        self.number_unicode_list = list(range(ord('0'), ord('9')+1))
        self.number_unicode_set = set(range(ord('0'), ord('9')+1))

    def get_sym_list(self):
        return self.symbol_unicode_list

    def get_alphabet_list(self):
        return self.alphabet_unicode_list

    def get_vietnam_list(self):
        return self.vietnam_unicode_list

    def get_number_list(self):
        return self.number_unicode_list

    def get_bg_sym_list(self):
        return self.bg_symbol_unicode_list

    def get_all_char_list(self):
        print('Charset.get_all_char_list')
        return self.symbol_unicode_list + self.alphabet_unicode_list + self.vietnam_unicode_list + self.number_unicode_list \
            + self.bg_symbol_unicode_list

    def get_all_char_set(self):
        return self.symbol_unicode_set | self.alphabet_unicode_set | self.vietnam_unicode_set | self.number_unicode_set \
            | self.bg_symbol_unicode_set

    def get(self, char):
        if self._is_alphabet(char):
            return "alphabet"
        elif self._is_num(char):
            return "number"
        elif self._is_symbol(char):
            return "symbol"
        elif self._is_signed_vnese(char):
            return "vietnam"
        else:
            return "unknown"

    def _is_signed_vnese(self, char):
        return ord(char) in self.vietnam_unicode_set

    def _is_alphabet(self, char):
        if ord('a') <= ord(char.lower()) <= ord('z'):
            return True
        else:
            return False

    def _is_num(self, char):
        if (ord('0') <= ord(char) <= ord('9')):
            return True
        else:
            return False

    def _is_symbol(self, char):
        if ord(char) in self.symbol_unicode_set:
            return True
        else:
            return False
