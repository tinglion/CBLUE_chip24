import json
import sys
import traceback

import logging

logger = logging.getLogger(__name__)


#
# ```json
# {
#   "送检时间（日期)": "2022-10-11",
#   "原始血细胞": "N/A",
#   "血片:原始粒细胞": "N/A",
#   "血片:嗜碱性粒细胞(嗜碱性中幼+嗜碱性晚幼+嗜碱性杆状核+嗜碱性分叶核四值总和)": "N/A"
# }
# ```
def cvt_str_to_obj(s):
    try:
        json_str = s

        posi_start = s.find("```json")
        if posi_start >= 0:
            posi_end = s.find("```", posi_start + 7)
            if posi_end >= 0:
                json_str = s[posi_start + 7 : posi_end]
        obj = json.loads(json_str)
        return obj
    except Exception as e:
        logger.error(f"sth wrong")
        traceback.print_exc()
    return None
