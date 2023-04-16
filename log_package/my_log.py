'''-*- coding: UTF-8 -*-
'''

import logging  # 导入longing模块


def log_to_file():
    '''
    create a log module,which can be called directly when using.
    '''
    logger = logging.getLogger("xaj_rizhi")  # 创建一个日志收集器，取名为xaj_rizhi,然后赋值给loggeer
    logger.setLevel(logging.INFO)  # 设置日志级别为info及以上
    fmt = "%(asctime)s  %(name)s  %(levelname)s  %(filename)s %(lineno)d %(message)s "  # 最终要看到文件名，行号，内容
    formatter = logging.Formatter(fmt)  # 实例化一个日志格式类，就是把日志输入的格式复制给formatter
    handle1 = logging.StreamHandler()  # 控制台（streamHandler） 这里要在控制台中显示，要取用logging模块的stramhandler
    handle1.setFormatter(formatter)
    logger.addHandler(handle1)
    handle12 = logging.FileHandler("log.txt", encoding="utf-8")
    handle12.setFormatter(formatter)
    logger.addHandler(handle12)
    return logger


if __name__ == '__main__':
    logger = log_to_file()
    logger.info('text log_package')
