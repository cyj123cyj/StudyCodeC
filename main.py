# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, './authorization')
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from interface.register import *
from interface.app import run_app
from interface.login_dialog import LoginDialog

#以下部分没用到，建议删掉
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig) #以上部分没用到


def login_dialog_main():
    app = QApplication(sys.argv)
    dialog = LoginDialog('医用CT成像设备质量检测系统--注册',
                         '用户名',
                         '说明：以上是您的机器码，请将此机器码提交给软件提供方，以获取授权注册码并输入',
                         '注册码',
                         '注册',
                         '取消')
    dialog.different_main()
    if dialog.exec_():
        run_app()


if __name__ == "__main__":
    reg = register()
    if reg.checkAuthored() == 1:
        run_app()
    else:
        login_dialog_main()
