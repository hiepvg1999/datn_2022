from collections import defaultdict
import re
# from starlette.responses import Response
import jellyfish
def post_processing(bboxes):
    result = defaultdict(str)
    bboxes.sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    check_flag_first_drugname = False
    result['drug_items'] = []
    type_boxes = defaultdict(list)
    for i, bbox in enumerate(bboxes):
        if bbox['pred'] != 'other':
            type_boxes[bbox['pred']].append(bbox)
    type_boxes['drugname'].sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    i = 0
    count = 0
    dic = defaultdict(set)
    while i < len(type_boxes['drugname'])-1:
        # print(type_boxes['drugname'][i]['text'])
        dic[count].add(type_boxes['drugname'][i]['text'])
        distance_nxt_bbox = type_boxes['drugname'][i+1]['bbox'][1] - type_boxes['drugname'][i]['bbox'][3]
        if distance_nxt_bbox <= 5 and 'Lời dặn BS' not in type_boxes['drugname'][i+1]['text']:
            dic[count].add(type_boxes['drugname'][i+1]['text'])
        else:
            count = count + 1
        i += 1
    
    type_boxes['quantity'].sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    type_boxes['usage'].sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    i = 0
    count = 0
    usage_dic = defaultdict(set)
    while i < len(type_boxes['usage'])-1:
        print(type_boxes['usage'][i]['text'])
        usage_dic[count].add(type_boxes['usage'][i]['text'])
        distance_nxt_bbox = type_boxes['usage'][i+1]['bbox'][1] - type_boxes['usage'][i]['bbox'][3]
        if distance_nxt_bbox <= 5:
            usage_dic[count].add(type_boxes['usage'][i+1]['text'])
        else:
            count = count + 1
        i += 1
    result['drug_items'] = []
    for i in range(len(type_boxes['quantity'])):
        item = {}
        item['drugname'] = ' '.join(reversed(tuple(dic[i])))
        item['quantity'] = type_boxes['quantity'][i]
        if 'Uống' in usage_dic[i] or 'Uong' in usage_dic[i]:
            item['usage'] = ' '.join(tuple(usage_dic[i]))
        else:
            item['usage'] = ' '.join(reversed(tuple(usage_dic[i])))
        print(dic[i])
        print(usage_dic[i])
        result['drug_items'].append(item)
        # src_y0 = bboxes[i]['bbox'][1]
        # src_y1 = bboxes[i]['bbox'][3]
        # item['drugname'] = bboxes[i]['text']
        # for j in type_boxes['quantity'] + type_boxes['usage']:
        #     nxt_y0 = bboxes[j]['bbox'][1]
        #     nxt_y1 = bboxes[j]['bbox'][3]
        #     if min(nxt_y1 - src_y0, src_y1 - nxt_y0) + 10 > 0:
        #         if bboxes[j]['text'].isdigit():
        #             item['quantity'] = bboxes[j]['text']
        #         else:
        #             item['usage'] = bboxes[j]['text']
    for i in range(len(bboxes)):
        # if bboxes[i]['pred'] != 'other':
        #     print(bboxes[i]['text'])
        if bboxes[i]['pred'] == 'drugname':
            check_flag_first_drugname = True
        if bboxes[i]['pred'] == 'date':
            result['date'] = bboxes[i]
        if bboxes[i]['pred'] == 'diagnose' and not check_flag_first_drugname:
            result['diagnose'] += (' ' + bboxes[i]['text'])
        
        
    index = result['diagnose'].find('đoán')
    if index != -1:
        result['diagnose'] = result['diagnose'][index-5:].strip()
    return result

def post_processing2(bboxes):
    result = defaultdict(list)
    bboxes.sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    for i in range(len(bboxes)):
        if bboxes[i]['pred'] != 'other':
            result[bboxes[i]['pred']].append(bboxes[i])
    return result

def post_processing3(bboxes):
    result = []
    bboxes.sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    for i in range(len(bboxes)):
        if bboxes[i]['pred'] != 'other':
            result.append(bboxes[i])
    return result

def post_processing4(bboxes):
    result = {}
    bboxes.sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    result['pills'] = []
    result['diagnose'] = ''
    visited = [0]*len(bboxes)
    # flag_first_drugname = False
    for idx, item in enumerate(bboxes):
        if jellyfish.jaro_distance(item['text'], 'Chuẩn đoán khác') > 0.8:
            visited[idx] = 1
        if item['pred'] == 'diagnose' and visited[idx] == 0:
            result['diagnose'] += ' ' + item['text']
            visited[idx] = 1
        if item['pred'] == 'other':
            continue
        if item['pred'] == 'drugname' and visited[idx] == 0:
            # flag_first_drugname = True
            check_end_pill= False
            pill = {}
            pill['usage'] = ''
            pill['name'] = re.sub(r"^[0-9]+['\- /]*[\)]\s*", '', item['text'])
            visited[idx] = 1
            pill['quantity'] =''
            for i in range(len(bboxes)):
                if bboxes[i]['pred'] == 'drugname' and visited[i] == 0 and i != idx:
                    break
                if not check_end_pill:
                    if bboxes[i]['pred'] == 'quantity' and visited[i] == 0:
                        pill['quantity'] += bboxes[i]['text'] + ' '
                        visited[i] = 1
                if bboxes[i]['pred'] == 'usage' and visited[i] == 0:
                    pill['usage'] += bboxes[i]['text'] + ' '
                    check_end_pill = True
                    visited[i] = 1
                    
            result['pills'].append(pill)
    if result['diagnose'] != '':
        if len(result['diagnose'].split(':')) >= 2:
            result['diagnose'] = result['diagnose'].split(':')[1]
        else:
            result['diagnose'] = result['diagnose']
    wrong_str = ['F00%','IIO','331','Ell', 'G46?', '110', '160', '170', '[25', '[10', '[20', '140', '1677','149','167t', '125', '150', '(((I10)', 'bàn I', '142', 'KS2', 'EI1', 'KO4', "167'", 'JII']
    true_str =  ['F00*','I10','J31','E11', 'G46*', 'I10', 'I60', 'I70', 'I25', 'I10', 'I20', 'J40','I67', 'I49', 'I67', 'I25', 'I50', '(I10)', 'bàn 1', 'J42', 'K52', 'E11', 'K04','I67','J11']
    for old, new in zip(wrong_str,true_str):
        result['diagnose'] = result['diagnose'].replace(old, new)
    return result

def post_processing5(bboxes):
    result = {}
    bboxes.sort(key= lambda x: (x['bbox'][1],x['bbox'][0]))
    result['pills'] = []
    result['diagnose'] = ''
    visited = [0]*len(bboxes)
    print("SAVE")
    for idx, item in enumerate(bboxes):
        if item['pred'] == 'diagnose':
            result['diagnose'] += ' ' + item['text']
            visited[idx] = 1
        if item['pred'] == 'other':
            continue
        if item['pred'] == 'drugname':
            pill = {}
            pill['usage'] = ''
            pill['name'] = re.sub(r"^[0-9]+['\- /]*[\)]\s*", '', item['text'])
            visited[idx] = 1
            pill['quantity'] =''
            
            print("Drugname: ", item)
            print("Used: ", [bboxes[i]['text'] for i in range(len(bboxes)) if visited[i] == 1])
            for i in range(len(bboxes)):
                if bboxes[i]['pred'] == 'drugname' and visited[i] and idx != i:
                    break
                if bboxes[i]['pred'] == 'quantity' and pill['quantity'] == '' and visited[i] == 0:
                    print("Quantity: ", bboxes[i])
                    pill['quantity'] += bboxes[i]['text'] + ' '
                    visited[i] = 1
                if bboxes[i]['pred'] == 'usage' and pill['usage'] == '' and visited[i] == 0:
                    print("Usage: ", bboxes[i])
                    visited[i] = 1
                    pill['usage'] += bboxes[i]['text'] + ' '
            result['pills'].append(pill)
    
    print()    
    print("BBOXES: ", bboxes)
    for b in bboxes:
        if b['pred'] != 'other':
            print(b)
    print("unused BBOXES: ")
    for i, b in enumerate(bboxes):
        if not visited[i] and bboxes[i]['pred'] != 'other':
            print(b)
    print()
    
    result['diagnose'] = result['diagnose'].split(':')[1]
    wrong_str = ['F00%','IIO','331','Ell', 'G46?', '110', '160', '170', '[25', '[10', '[20', '140', '1677','149','167t', '125', '150', '(((I10)', 'bàn I', '142', 'KS2', 'EI1', 'KO4', "167'", 'JII']
    true_str =  ['F00*','I10','J31','E11', 'G46*', 'I10', 'I60', 'I70', 'I25', 'I10', 'I20', 'J40','I67', 'I49', 'I67', 'I25', 'I50', '(I10)', 'bàn 1', 'J42', 'K52', 'E11', 'K04','I67','J11']
    for old, new in zip(wrong_str,true_str):
        result['diagnose'] = result['diagnose'].replace(old, new)
    return result
    
    