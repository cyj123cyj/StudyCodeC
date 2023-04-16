import sys
from interface.languagecheck import languageok
# Try imports and report missing packages.
# error = False
# Just to check presence of essential libraries
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
from interface.viewer import Viewer


def run_app():
    if len(sys.argv) < 2:
        path = ".."
    else:
        path = sys.argv[1]
    app = QApplication(sys.argv)
    lan = languageok()
    print(lan.check_language())
    if lan.check_language() == 1:
        trans = QtCore.QTranslator()
        trans.load("ctdicom_en")
        app.installTranslator(trans)
        print(1)
    elif lan.check_language() == 0:#是不是可以整合到一起，只有print没有其他功能
        print(2)
    elif lan.check_language() == -6:
        print(3)
    elif lan.check_language() == -10:
        print(4)
    scaleRate = app.screens()[0].logicalDotsPerInch() / 96
    font = QFont()
    font.setPixelSize(int(14 * scaleRate))  # *scaleRate
    app.setFont(font)
    viewer = Viewer(path)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
