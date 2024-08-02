import json
import logging
import os
import random
import traceback
import zipfile

logger = logging.getLogger(__name__)

upperLetter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowerLetter = "abcdefghigklmnopqrstuvwxyz"
digits = "0123456789"
wpecialCharacters = "!@#$%&_-.+="


def getRandomString(base, randomlength=12):
    """
    生成一个指定长度的随机字符串
    """
    str_list = [random.choice(base) for i in range(randomlength)]
    random_str = "".join(str_list)
    return random_str


def getRandomStringByMode(mode="mixDigitLetter", len=12):
    # 按照不同模式生成随机字符串
    randomMap = {
        "digit": digits,
        "upper": upperLetter,
        "lower": lowerLetter,
        "mixDigitLetter": upperLetter + lowerLetter + digits,
        "mixLetter": upperLetter + lowerLetter,
        "mixDigitLetterCharcter": upperLetter
        + lowerLetter
        + digits
        + wpecialCharacters,
    }
    return getRandomString(randomMap[mode], len)


def zip_folder(folder_path, zip_path):
    """
    压缩文件夹

    :param folder_path: 要压缩的文件夹路径
    :param zip_path: 压缩文件的保存路径
    """
    # 创建一个 ZipFile 对象来保存压缩文件
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 构造文件的完整路径
                file_path = os.path.join(root, file)
                # 构造文件在压缩文件中的相对路径
                rel_path = os.path.relpath(file_path, os.path.dirname(folder_path))
                # 将文件添加到压缩文件中
                zipf.write(file_path, rel_path)
        zipf.close()


def getExtention(path):
    _, extension = os.path.splitext(path)
    if len(extension) > 0:
        return extension[1:]
    else:
        return ""


def write_content(content, fn):
    with open(fn, "w", encoding="utf8") as fp:
        fp.write(content)
        fp.close()


def write_bytes(content, fn):
    with open(fn, "wb") as fp:
        fp.write(content)
        fp.close()


def is_number(num):
    try:
        float(num)
        return True
    except ValueError:
        pass

    try:
        import unicodedata

        unicodedata.numeric(num)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_string_len(string):
    """
    获取字符串最大长度
    :param string:
    :return:
    """
    length = 4
    if string is None:
        return length
    if is_number(string):
        return length
    for char in string:
        length += 2.1 if ord(char) > 256 else 1
    return round(length, 1)


if __name__ == "__main__":
    print(f"{get_string_len(24332)}")
    # download_file(
    #     "http://192.168.19.146:9031/med/AI%20is%20about%20to%20completely%20change%20how%20you%20use%20computers%20_%20Bill%20Gates.pdf",
    #     "temp/ai.pdf",
    # )
