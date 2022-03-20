"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Class for loading Meta information, such as names of files with masks, list of categories, label <-> category matching, etc.
"""

import os
import re
import random
import csv
import settings
import numpy as np
import formulas.formula as F
from collections import OrderedDict
from functools import partial

import S
import util.information_printer

categories = []
map_label_name_2_label = {}
map_label_2_label_name = {}
map_image_i_2_image_filename = {}
metadata_raw = []
map_cat_global_i_2_local_i = {}
map_cat_local_i_2_global_i = {}
map_label_2_max_coverage_c_i = []


def init():
    """ Loads the values of the following global variables, which should all be self-explanatory."""
    global map_label_name_2_label, map_label_2_label_name, map_image_i_2_image_filename, map_cat_global_i_2_local_i,\
        map_cat_local_i_2_global_i, map_label_2_max_coverage_c_i, categories, metadata_raw

    def decode_index_dict(row):
        result = {}
        for key, val in row.items():
            if key in ["image", "split"]:
                result[key] = val
            elif key in ["sw", "sh", "iw", "ih"]:
                result[key] = int(val)
            else:
                item = [s for s in val.split(";") if s]
                for i, v in enumerate(item):
                    if re.match("^\d+$", v):
                        item[i] = int(v)
                result[key] = item
        return result

    def decode_label_dict(row_in_label_csv):
        result = {}
        for key, val in row_in_label_csv.items():
            if key == "category":
                result[key] = dict((c, int(n)) for c, n in [re.match("^([^(]*)\(([^)]*)\)$", f).groups() for f in val.split(";")])
            elif key == "name":
                result[key] = val
            elif key == "syns":
                result[key] = val.split(";")
            elif re.match("^\d+$", val):
                result[key] = int(val)
            elif re.match("^\d+\.\d*$", val):
                result[key] = float(val)
            else:
                result[key] = val
        return result

    def build_numpy_category_map(map_data, key_local_i="code", key_global_i="number"):
        map_local_2_global = {}
        map_global_2_local = {}

        for i, row in enumerate(map_data):
            # The local id is simply i
            map_local_2_global[i] = row[key_local_i] - 1
            map_global_2_local[row[key_global_i] - 1] = i

        return map_local_2_global, map_global_2_local

    def build_dense_label_array(label_data):
        """
        Input: set of rows with 'number' fields (or another field name key).
        Output: array such that a[number] = the row with the given number.
        """
        labels_as_array = [None] * len(label_data)
        for i, row in enumerate(label_data):
            labels_as_array[i] = row

        return labels_as_array

    label_data = []

    with open(os.path.join(settings.DATA_DIRECTORY, settings.INDEX_FILE)) as file: ## index_ade20k.csv
        metadata_raw = [decode_index_dict(r) for r in csv.DictReader(file)]


    with open(os.path.join(settings.DATA_DIRECTORY, "category.csv")) as file:
        map_catname_2_metadata = OrderedDict()
        map_cati_2_catname = {}
        for (i, row) in enumerate(csv.DictReader(file)):
            map_catname_2_metadata[row["name"]] = i
            map_cati_2_catname[i] = row["name"]


    categories = list(map_catname_2_metadata.keys())


    with open(os.path.join(settings.DATA_DIRECTORY, "label.csv")) as file:
        # stores in which categories a label appears (mostly but not always just one)
        label_data = build_dense_label_array([decode_label_dict(r) for r in csv.DictReader(file)])

    # -------------------------------------------------------------------------------------------------
    # - Immediately kill all _ characters and replace them with whitespaces to have a unified format. -
    # -------------------------------------------------------------------------------------------------
    for i in range(len(label_data)):
        label_data[i]["name"] = label_data[i]["name"].replace('_', ' ')




    map_label_name_2_label = {row["name"] : (i + 1) for i, row in enumerate(label_data)}
    map_label_2_label_name = {(i + 1) : row["name"] for i, row in enumerate(label_data)}

    def index_has_any_data(row_, categories_):
        for c in categories_:
            for data in row_[c]:
                if data:
                    return True
        return False

    # Filter out images with insufficient segmentation_data
    filter_function = partial(index_has_any_data, categories_=categories)
    metadata_raw = [row for row in metadata_raw if filter_function(row)]
    map_image_i_2_image_filename = [row["image"] for row in metadata_raw]


    # Build dense remapping arrays for labels, so that you can get dense ranges of labels for each dict_catname_to_catid.
    map_cat_global_i_2_local_i = {} # element at x is a map from the global to the local id for labels corresponding to category x
    map_cat_local_i_2_global_i = {} # as above in reverse
    categories_metadata = {}
    map_catname_2_cat_label_data = {}

    # for cat in self.dict_catname_to_catid:
    for category in categories:
        with open(os.path.join(settings.DATA_DIRECTORY, "c_%s.csv" % category)) as file:
            c_data = [decode_label_dict(r) for r in csv.DictReader(file)]
        map_cat_local_i_2_global_i[category], map_cat_global_i_2_local_i[category] = build_numpy_category_map(c_data)
        categories_metadata[category] = build_dense_label_array(c_data)
        map_catname_2_cat_label_data[category] = c_data

    S.n_labels = len(label_data)
    S.n_categories = len(map_catname_2_metadata.keys())
    S.n_images = len(metadata_raw)

    util.information_printer.show_single(f"Categories in order: {categories}")
    coverage = (lambda category,j : label_data[j]["coverage"] if category is None else categories_metadata[category][j]["coverage"])

    M = map_cat_global_i_2_local_i
    map_label_2_max_coverage_c_i = [0]  # To make it so entries start at 1
    for label_i in range(S.n_labels):
        label = label_i + 1
        category_coverages = [coverage(C, M[C][label_i]) if label_i in M[C] else 0 for C in categories]
        # category_coverages = [self.coverage(C, M[C][label_i]) for C in categories]
        if settings.PRINT_INFORMATION and random.random() < 0.004:
            print(f"Inspecting label #{label}, {name(label)}: coverages = {[f'{C:.1f}' for C in category_coverages]} -> chosen = #{np.argmax(category_coverages)}")

        map_label_2_max_coverage_c_i.append(np.argmax(category_coverages))
    if settings.PRINT_INFORMATION:
        print("\n")

    F.label_namer = name

def name(label_or_formula):
    """
    Returns an English name for the given label.
    """
    label = label_or_formula if type(label_or_formula) is int else label_or_formula.val

    return map_label_2_label_name[label]

def load_csv(filename, readfields=None):
    def convert(value):
        if re.match(r"^-?\d+$", value):
            try:
                return int(value)
            except:
                pass
        if re.match(r"^-?[\.\d]+(?:e[+=]\d+)$", value):
            try:
                return float(value)
            except:
                pass
        return value

    with open(filename) as f:
        reader = csv.DictReader(f)
        result = [{k: convert(v) for k, v in row.items()} for row in reader]
        if readfields is not None:
            readfields.extend(reader.fieldnames)
    return result

def filename(image_i):
    """The filename of the ith jpeg (original image)."""
    return os.path.join(settings.DATA_DIRECTORY, "images", map_image_i_2_image_filename[image_i])


