import json


def convert(data_json):
    result = []
    for r in data_json:
        result.append(
            "%s@%s@%s@%s@%s"
            % (
                r.get("案例编号", ""),
                r.get("predict_1", ""),
                r.get("predict_2", ""),
                r.get("predict_3", ""),
                r.get("predict_4", ""),
            )
        )
    return result


def convert_file(fn_src, fn_dst):
    with open(fn_src, "r", encoding="utf8") as fp_src:
        data = json.load(fp_src)
        fp_src.close()
        result = convert(data)
        with open(fn_dst, "w", encoding="utf8") as fp_dst:
            for l in result:
                fp_dst.write(f"{l}\n")
            fp_dst.close()


if __name__ == "__main__":
    convert_file(
        fn_src="./temp/NE_B_44.json",
        fn_dst="./temp/NE_B_44.txt",

        # fn_src="./temp/NE_train_44s2_180_190.json",
        # fn_dst="./temp/NE_train_44_180_190.txt",

        # fn_src="./temp/NE_A_44_0_50.json",
        # fn_dst="./temp/NE_A_44_0_50.txt",
    )
