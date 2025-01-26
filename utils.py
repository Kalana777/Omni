from transformers import BertTokenizer
from io import open
import torch
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
import config


def kdelta_encode_polygon(poly, original_k):
    polycoding = []
    num_poly_verts = len(poly)
    k = min(original_k, max(num_poly_verts // 2 - 1, 1))

    for i in range(0, num_poly_verts):
        tmp = np.array(poly[i])
        tmp1 = np.expand_dims(tmp, axis=0)
        tmp = np.tile(tmp1, (k * 2, 1))
        if i == 0:
            neis = np.concatenate([np.array(poly[i - k:]), np.array(poly[i + 1:i + 1 + k])])
        elif i == num_poly_verts - 1:
            neis = np.concatenate([np.array(poly[i - k:i]), np.array(poly[:i + k - num_poly_verts + 1])])
        elif i < k and i + k < num_poly_verts:
            neis = np.concatenate([np.array(poly[i - k:]), np.array(poly[:i]), np.array(poly[i + 1:i + 1 + k])])
        elif i >= k and i + k >= num_poly_verts:
            neis = np.concatenate(
                [np.array(poly[i - k:i]), np.array(poly[i + 1:]), np.array(poly[:i + k - num_poly_verts + 1])])
        else:
            neis = np.concatenate([np.array(poly[i - k:i]), np.array(poly[i + 1:i + 1 + k])])
        tmp = tmp - neis
        if tmp.shape[0] != original_k * 2:
            tmp = np.pad(tmp, ((0, original_k * 2 - tmp.shape[0]), (0, 0)), mode='constant', constant_values=0)
        try:
            tmp = np.concatenate([tmp1, tmp], axis=0)
        except:
            print(tmp1.shape, tmp1, poly[i])
        tmp = np.expand_dims(np.concatenate(tmp, axis=0), axis=0)
        polycoding.append(tmp)

    polycoding = np.concatenate(polycoding, axis=0)
    return polycoding


def kdelta_encode_line(poly, original_k):
    polycoding = []
    num_poly_verts = len(poly)
    k = min(original_k, max(num_poly_verts // 2 - 1, 1))

    for i in range(0, num_poly_verts):
        tmp = np.array(poly[i])
        tmp1 = np.expand_dims(tmp, axis=0)
        tmp = np.tile(tmp1, (k * 2, 1))

        if i == 0:
            neis = np.concatenate([np.tile(tmp1, (k - i, 1)), np.array(poly[i + 1:i + 1 + k])])
        elif i == num_poly_verts - 1:
            neis = np.concatenate([np.array(poly[i - k:i]), np.tile(tmp1, (k, 1))])
        elif i < k and i + k < num_poly_verts:
            neis = np.concatenate([np.tile(tmp1, (k - i, 1)), np.array(poly[:i]), np.array(poly[i + 1:i + 1 + k])])
        elif i >= k and i + k >= num_poly_verts:
            neis = np.concatenate(
                [np.array(poly[i - k:i]), np.array(poly[i + 1:]), np.tile(tmp1, (k - num_poly_verts + i + 1, 1))])
        else:
            neis = np.concatenate([np.array(poly[i - k:i]), np.array(poly[i + 1:i + 1 + k])])
        tmp = tmp - neis
        if tmp.shape[0] != original_k * 2:
            tmp = np.pad(tmp, ((0, original_k * 2 - tmp.shape[0]), (0, 0)), mode='constant', constant_values=0)
        tmp = np.concatenate([tmp1, tmp], axis=0)
        tmp = np.expand_dims(np.concatenate(tmp, axis=0), axis=0)
        polycoding.append(tmp)

    polycoding = np.concatenate(polycoding, axis=0)
    return polycoding


def make_geom_dataset(geo_list, geo_type_list, k):
    dataset = []

    for poly, type_enc in zip(geo_list, geo_type_list):

        poly_type = read_poly_type_onehot(type_enc)

        # if point, polygon or multipolygon --> do this...

        if poly_type in ['point', 'poly', 'multipoly']:
            if len(poly) == 300:
                polycoding = kdelta_encode_polygon(poly, k)

            if len(poly) != 300:
                polycoding = []
                for small_pol in poly:
                    small_pol_coding = kdelta_encode_polygon(small_pol, k)
                    polycoding.append(small_pol_coding)
                polycoding = np.concatenate(polycoding, axis=0)

        else:
            if len(poly) == 300:
                polycoding = kdelta_encode_line(poly, k)
            if len(poly) != 300:
                polycoding = []
                for small_pol in poly:
                    small_pol_coding = kdelta_encode_line(small_pol, k)
                    polycoding.append(small_pol_coding)
                polycoding = np.concatenate(polycoding, axis=0)
        dataset.append(polycoding)

    dataset = np.array(dataset)

    return torch.FloatTensor(dataset).permute(0, 2, 1), torch.FloatTensor(geo_type_list)


def compute_dist(lat1, lon1, lat2, lon2):
    R = 6373.0

    try:
        float(lat1)
    except ValueError:
        return -1

    try:
        float(lon1)
    except ValueError:
        return -1

    try:
        float(lat2)
    except ValueError:
        return -1

    try:
        float(lon2)
    except ValueError:
        return -1

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))

    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    dist = round(R * c * 1000)
    dist = 2 * (dist) / config.MAX_DIST - 1

    return dist


def get_lat_long(entity):
    # words = entity.lower().split()
    words = entity.split()
    for i, word in enumerate(words):
        if words[i - 2] == 'latitude' and words[i - 1] == 'VAL':
            latitude = float(word)
            longitude = float(words[i + 4])
            idx = i

        if words[i - 2] == 'postalcode' and words[i - 1] == 'VAL':
            try:
                words[i] = str(int(float(words[i])))
            except ValueError:
                pass

    del words[idx - 3:idx + 5]

    return ' '.join(words), latitude, longitude


def get_geom_list(df, col_name, g_typ_name):
    g_list = list(df[col_name])
    g_type_list = list(df[g_typ_name])

    fin_list = []

    for g in g_list:

        if isinstance(g, Polygon):
            tmp = g.exterior.coords[:-1]
            fin_list.append(tmp)

        elif isinstance(g, LineString):
            tmp = g.coords[:]
            fin_list.append(tmp)

        elif isinstance(g, MultiPolygon):
            tmp = [geom.exterior.coords[:-1] for geom in g.geoms]
            fin_list.append(tmp)

        elif isinstance(g, MultiLineString):
            tmp = [geom.coords[:] for geom in g.geoms]
            fin_list.append(tmp)

    return fin_list, g_type_list, g_list


def read_poly_type_onehot(onehot):
    poly_types = ['point', 'poly', 'multipoly', 'line', 'multiline']
    result = poly_types[np.argmax(onehot)]
    return result


def calculate_min_distance(geoms_left, geoms_right):
    min_dists = []
    for l, r in zip(geoms_left, geoms_right):
        min_dist = l.distance(r) / (2 * sqrt(2))
        min_dists.append(min_dist)
    return min_dists


def prepare_dataset(path, geom_path, attributes, run_att_aff, max_seq_len=128, k=12):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # special_tokens_dict = {'additional_special_tokens': ['[VAL]']}
    # num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    data_x = []
    coord_x = []
    data_y = []
    val_token_positions = []

    # cos_token_positions = {"names":[], "types":[], "addresses":[]}
    # cos_token_positions = {"names":[], "addresses":[]}
    cos_token_positions = {'attribute1': [], 'attribute2': []}
    line_count = 0
    with open(path, 'r', encoding='utf-8') as f:

        for line in f:

            if attributes[0] in line and attributes[1] in line:
                pass
            elif run_att_aff:
                print('Please provide attributes that are present in the dataset. For example, \'name type\' for NZER')
                quit()
            arr = line.split('\t')
            y = arr[2]
            e1, lat1, long1 = get_lat_long(arr[0])
            e2, lat2, long2 = get_lat_long(arr[1])

            if len(arr) > 2:

                x = tokenizer.tokenize('[CLS] ' + e1 + ' [SEP] ' + e2 + ' [SEP]')
                # val_tokens = [i for i, token in enumerate(x) if token == "[VAL]"]
                val_tokens = []
                for i, token in enumerate(x):
                    if token == "val":
                        if x[i - 1] in ['name', 'type', 'address', 'code']:
                            val_tokens.append(i)

                entity1_name_start, entity1_name_end, entity2_name_start, entity2_name_end = None, None, None, None

                for idx in range(len(x) - 2):
                    # if x[idx] == 'col' and x[idx + 1] == 'name' and x[idx + 2] == 'val':
                    if x[idx] == 'col' and x[idx + 1] == attributes[0] and x[idx + 2] == 'val':
                        name_start = idx + 3
                        # Find the next 'col' token
                        name_end = name_start
                        while name_end < len(x) and not (
                                x[name_end] in ['col', '[SEP]'] and x[name_end + 1] in ['name', 'type', 'address',
                                                                                        'postal']):
                            name_end += 1
                        if entity1_name_start is None:
                            # Entity 1
                            entity1_name_start = name_start
                            entity1_name_end = name_end
                        else:
                            # Entity 2
                            entity2_name_start = name_start
                            entity2_name_end = name_end
                if entity1_name_start == entity1_name_end or entity2_name_start == entity2_name_end:
                    print(x)
                # cos_token_positions['names'].append((entity1_name_start, entity1_name_end, entity2_name_start, entity2_name_end, x[entity1_name_start:entity1_name_end], x[entity2_name_start:entity2_name_end]))
                cos_token_positions['attribute1'].append((entity1_name_start, entity1_name_end, entity2_name_start,
                                                          entity2_name_end, x[entity1_name_start:entity1_name_end],
                                                          x[entity2_name_start:entity2_name_end]))

                # entity1_name_start, entity1_name_end,  entity2_name_start, entity2_name_end= None, None, None, None
                # for idx in range(len(x) - 2):
                #     if x[idx] == 'col' and x[idx + 1] == 'address' and x[idx + 2] == 'val':
                #         name_start = idx + 3
                #         # Find the next 'col' token
                #         name_end = name_start
                #         while name_end < len(x) and not (x[name_end] in ['col', '[SEP]'] and x[name_end+1] in ['name','type','address', 'postal']):
                #             name_end += 1
                #         if entity1_name_start is None:
                #             # Entity 1
                #             entity1_name_start = name_start
                #             entity1_name_end = name_end
                #         else:
                #             # Entity 2
                #             entity2_name_start = name_start
                #             entity2_name_end = name_end
                # # if entity1_name_start==entity1_name_end or entity2_name_start==entity2_name_end:
                #     # print(x)
                # cos_token_positions['addresses'].append(
                #     (entity1_name_start, entity1_name_end, entity2_name_start, entity2_name_end, x[entity1_name_start:entity1_name_end], x[entity2_name_start:entity2_name_end]))

                entity1_name_start, entity1_name_end, entity2_name_start, entity2_name_end = None, None, None, None

                for idx in range(len(x) - 2):
                    # if x[idx] == 'col' and x[idx + 1] == 'type' and x[idx + 2] == 'val':
                    if x[idx] == 'col' and x[idx + 1] == attributes[1] and x[idx + 2] == 'val':
                        name_start = idx + 3
                        # Find the next 'col' token
                        name_end = name_start
                        while name_end < len(x) and not (x[name_end] in ['[SEP]']):
                            name_end += 1
                        if entity1_name_start is None:
                            # Entity 1
                            entity1_name_start = name_start
                            entity1_name_end = name_end
                        else:
                            # Entity 2
                            entity2_name_start = name_start
                            entity2_name_end = name_end
                if entity1_name_start == entity1_name_end or entity2_name_start == entity2_name_end:
                    print(x)
                # cos_token_positions['addresses'].append((entity1_name_start, entity1_name_end, entity2_name_start, entity2_name_end, x[entity1_name_start:entity1_name_end], x[entity2_name_start:entity2_name_end]))
                cos_token_positions['attribute2'].append((entity1_name_start, entity1_name_end, entity2_name_start,
                                                          entity2_name_end, x[entity1_name_start:entity1_name_end],
                                                          x[entity2_name_start:entity2_name_end]))

                y = arr[2]

                if len(x) < max_seq_len:
                    padded_x = x + ['[PAD]'] * (max_seq_len - len(x))
                else:
                    padded_x = x[:max_seq_len]

                data_x.append(tokenizer.convert_tokens_to_ids(padded_x))
                coord_x.append(compute_dist(lat1, long1, lat2, long2))
                data_y.append(int(y.strip()))
                val_token_positions.append(val_tokens)
                # cos_token_positions.append(cos_tokens)

            line_count += 1

    geoms = pd.read_pickle(geom_path)
    geoms_left, type_left, g_shapely_left = get_geom_list(geoms, 'norm_left', 'left_type')
    geoms_right, type_right, g_shapely_right = get_geom_list(geoms, 'norm_right', 'right_type')

    min_dists_x = calculate_min_distance(g_shapely_left, g_shapely_right)

    geoms_left, type_left = make_geom_dataset(geoms_left, type_left, k // 2)
    geoms_right, type_right = make_geom_dataset(geoms_right, type_right, k // 2)

    geoms = {'geoms_left': geoms_left, 'type_left': type_left, 'geoms_right': geoms_right, 'type_right': type_right}

    return data_x, coord_x, data_y, geoms, min_dists_x, val_token_positions, cos_token_positions


