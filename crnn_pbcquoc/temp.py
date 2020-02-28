import json

with open('temp.json','w',encoding='utf-8') as f:
    json.dump('Đường Nguyễn Phong Sắc, Huyện Thủy Nguyên, Hải Phòng, Đường Lâm Hạ, Quận Long Biên, Hà Nội,  ̀'  , f, ensure_ascii=True)