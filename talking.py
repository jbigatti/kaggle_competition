"""
Extract data from datasets.

Usage:
    talking.py [options]
    talking.py -h | --help

Options:
  -h --help                     Show this screen.
  --test                        To generate "full_test_dataset.pkl".
                                Default generate "full_trian_dataset.pkl"
"""

import csv
import pickle
import logging
from docopt import docopt

logger = logging.getLogger(__name__)

# event_id app_id is_installed is_active
APP_EVENTS_CSV = "talkingdata/app_events.csv"

# event_id,device_id,timestamp,longitude,latitude
EVENTS_CSV = "talkingdata/events.csv"

# label_id, category
LABEL_CATEGORIES_CSV = "talkingdata/label_categories.csv"

# app_id,label_id
APP_LABELS_CSV = "talkingdata/app_labels.csv"

# device_id,phone_brand,device_model
PHONE_BRAND_DEVICE_MODEL_CSV = "talkingdata/phone_brand_device_model.csv"

# device_id,gender,age,group
GENDER_AGE_TRAIN_CSV = "talkingdata/gender_age_train.csv"

GENDER_AGE_TEST_CSV = "talkingdata/gender_age_test.csv"


PICKLE_TRAIN_NAME = "full_train_dataset.pkl"
PICKLE_TEST_NAME = "full_test_dataset.pkl"


"""
Example of devide representation

{
    "device_id": 5646546546,
    "brand": "una marca",
    "model": "un modelo",
    # Event list of this device.
    "apps_id": [installed app_id],
    "labels": [installed app_labels],
    "group": target,
}
"""

BRAND_TRANSLATE = {
    "UNKOWN": "UNKWON",
    "三星": "samsung",
    "天语": "Ktouch",
    "海信": "hisense",
    "联想": "lenovo",
    "欧比": "obi",
    "爱派尔": "ipair",
    "努比亚": "nubia",
    "优米": "youmi",
    "朵唯": "dowe",
    "黑米": "heymi",
    "锤子": "hammer",
    "酷比魔方": "koobee",
    "美图": "meitu",
    "尼比鲁": "nibilu",
    "一加": "oneplus",
    "优购": "yougo",
    "诺基亚": "nokia",
    "糖葫芦": "candy",
    "中国移动": "ccmc",
    "语信": "yuxin",
    "基伍": "kiwu",
    "青橙": "greeno",
    "华硕": "asus",
    "夏新": "panosonic",
    "维图": "weitu",
    "艾优尼": "aiyouni",
    "摩托罗拉": "moto",
    "乡米": "xiangmi",
    "米奇": "micky",
    "大可乐": "bigcola",
    "沃普丰": "wpf",
    "神舟": "hasse",
    "摩乐": "mole",
    "飞秒": "fs",
    "米歌": "mige",
    "富可视": "fks",
    "德赛": "desci",
    "梦米": "mengmi",
    "乐视": "lshi",
    "小杨树": "smallt",
    "纽曼": "newman",
    "邦华": "banghua",
    "E派": "epai",
    "易派": "epai",
    "普耐尔": "pner",
    "欧新": "ouxin",
    "西米": "ximi",
    "海尔": "haier",
    "波导": "bodao",
    "糯米": "nuomi",
    "唯米": "weimi",
    "酷珀": "kupo",
    "谷歌": "google",
    "昂达": "ada",
    "聆韵": "lingyun"}


def add_events_ids(result):
    with open(EVENTS_CSV) as event_file:
        reader = csv.DictReader(event_file)
        for row in reader:
            device_id = str(row["device_id"])
            event_id = str(row["event_id"])
            device = result.get(device_id, None)
            if not device:
                continue
            else:
                old = device.get("events_id", None)
                if old is None:
                    device["events_id"] = set([event_id])
                else:
                    device["events_id"].add(event_id)
            result[device_id] = device
    return result


def app_event_dict():
    result = {}
    with open(APP_EVENTS_CSV) as app_event_file:
        reader = csv.DictReader(app_event_file)
        for row in reader:
            event_id, app_id = str(row["event_id"]), row["app_id"]
            old = result.get(event_id, None)
            if old is None:
                result[event_id] = set([app_id])
            else:
                result[event_id].add(app_id)
    return result


def add_app_events(result):
    app_event = app_event_dict()
    for k, v in result.items():
        new_v = []
        for x in v.get("events_id", []):
            for y in app_event.get(x, []):
                new_v.append(y)
        result[k]["apps_id"] = set(new_v)
        v.pop("events_id", None)
    return result


def get_app_labels():
    app_labels = {}
    with open(APP_LABELS_CSV) as app_labels_file:
        reader = csv.DictReader(app_labels_file)
        for row in reader:
            # app_id,label_id
            app_id = str(row["app_id"])
            label_id = str(row["label_id"])
            old = app_labels.get(app_id, None)
            if old is None:
                app_labels[app_id] = set([label_id])
            else:
                app_labels[app_id].add(label_id)
    return app_labels


def get_label_categories():
    label_cateogries = {}
    with open(LABEL_CATEGORIES_CSV) as label_categories_file:
        reader = csv.DictReader(label_categories_file)
        for row in reader:
            # label_id, category
            label_cateogries[str(row["label_id"])] = row["category"]
    return label_cateogries


def merge_app_with_labels():
    app_labels = get_app_labels()
    label_cateogries = get_label_categories()
    for app_id, label_ids in app_labels.items():
        labels_as_string = []
        for label_id in label_ids:
            labels_as_string.append(label_cateogries[label_id])
        app_labels[app_id] = labels_as_string
    return app_labels


def add_app_labels(result):
    app_labels = merge_app_with_labels()
    for k, v in result.items():
        labels = []
        for x in v["apps_id"]:
            for y in app_labels.get(str(x), []):
                labels.append(y)
        result[k]["labels"] = labels
    return result


def add_phone_brand(result):
    with open(PHONE_BRAND_DEVICE_MODEL_CSV) as phone_brand_device_model_file:
        reader = csv.DictReader(phone_brand_device_model_file)
        for row in reader:
            device_id = str(row["device_id"])
            device_brand = BRAND_TRANSLATE.get(row["phone_brand"], "UNKOWN")
            device_model = row["device_model"]
            old = result.get(device_id, None)
            if old is None:
                continue
            else:
                result[device_id]["brand"] = device_brand
                result[device_id]["model"] = device_model
    return result


# device_id,gender,age,group
def add_gender_age_group(train=True):
    result = {}
    if train:
        file_name = GENDER_AGE_TRAIN_CSV
    else:
        file_name = GENDER_AGE_TEST_CSV

    with open(file_name) as gender_age_train_file:
        reader = csv.DictReader(gender_age_train_file)
        for row in reader:
            device_id = str(row["device_id"])
            if train:
                result[device_id] = {
                    "device_id": device_id,
                    "group": row["group"],
                    "gender": row["gender"],
                    "age": row["age"]}
            else:
                result[device_id] = {"device_id": device_id}
    return result


def main(train=True):
    if train:
        file_name = PICKLE_TRAIN_NAME
    else:
        file_name = PICKLE_TEST_NAME
    pre_result = add_gender_age_group(train)
    add_phone_brand(pre_result)
    add_events_ids(pre_result)
    add_app_events(pre_result)
    add_app_labels(pre_result)
    result = []
    for k, v in pre_result.items():
        result.append(v)
    pickle.dump(result, open(file_name, "wb"))
    return result


if __name__ == "__main__":
    opts = docopt(__doc__)
    train = True
    if opts["--test"]:
        train = False
    main(train=train)
