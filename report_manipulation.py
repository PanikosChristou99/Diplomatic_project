from io import StringIO
import json
from typing import Sequence
import pandas
import re

names = ["item_name", "precision", "recall", "f1_score", "support"]


def parse_rep(lines: Sequence[str]) -> dict:

    start_lines = []
    end_lines = []

    for line in lines[2:12]:
        line = '   '+line

        line = re.sub('[^A-Za-z0-9] +[^A-Za-z0-9]', '\t', line)
        start_lines.append(line)

    # string = "\n".join(line[1:-1] for line in start_lines)

    # # print(string)

    # detentions = StringIO(string)

    df = pandas.read_csv(StringIO("\n".join(line[1:-1] for line in start_lines)), sep='\t', index_col=None,
                         names=names, header=None)

    # print(df)

    # print('-----------------')

    for line in lines[13:15]:
        line = '   '+line

        line = re.sub('[^A-Za-z0-9] +[^A-Za-z0-9]', '\t', line)
        end_lines.append(line)

        # print(line)

    df2 = pandas.read_csv(StringIO("\n".join(line[1:-1] for line in end_lines)), sep='\t', index_col=None,
                          names=names, header=None)

    df3 = df.append(df2)
    # print(df)

    ret_dict = {}

    ret_dict['item_names'] = df["item_name"].tolist()
    ret_dict['precision'] = df3["precision"].tolist()
    ret_dict['recall'] = df3["recall"].tolist()
    ret_dict['f1_score'] = df3["f1_score"].tolist()
    ret_dict['support'] = df["support"].tolist()

    return(ret_dict)


f = open("rep22_46_06_03.txt", "r")

lines = f.readlines()

print(json.dumps(parse_rep(lines), indent=4))

df = pandas.read_csv(
    r'.\volume\edges\stats\19_21_07_03_edge1_requests_fasterrcnn_mobilenet_v3_large_fpn_BW_1_resize_25%_quality_25%.csv', index_col=0)

print(df)
