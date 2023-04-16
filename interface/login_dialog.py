import sys

sys.path.insert(0, './authorization')
from PyQt5.QtWidgets import *
from interface.register import *
from PyQt5.QtCore import Qt
from register_code.encrypter import encrypter


class LoginDialog(QDialog):
    def __init__(self, title, username, plaintext, codetext, pblogin, canceltext, parent=None):
        QDialog.__init__(self, parent, flags=None)
        self.same(title, username, plaintext, codetext, pblogin, canceltext)

    def same(self, title, username, plaintext, codetext, pblogin, canceltext):
        self.setWindowTitle(title)
        self.resize(400, 150)
        self.leName = QLineEdit(self)
        self.leName.setPlaceholderText(username)
        self.plainTextEdit = QPlainTextEdit(self)
        self.plainTextEdit.setPlainText(plaintext)
        self.lePassword = QLineEdit(self)
        self.lePassword.setPlaceholderText(codetext)
        self.pbLogin = QPushButton(pblogin, self)
        self.pbCancel = QPushButton(canceltext, self)
        self.pbCancel.clicked.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(self.leName, alignment=Qt.Alignment())
        layout.addWidget(self.plainTextEdit, alignment=Qt.Alignment())
        layout.addWidget(self.lePassword, alignment=Qt.Alignment())
        spacerItem = QSpacerItem(20, 48, QSizePolicy.Minimum, QSizePolicy.Expanding)  # 放一个间隔对象美化布局
        layout.addItem(spacerItem)
        buttonLayout = QHBoxLayout()
        spancerItem2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        buttonLayout.addItem(spancerItem2)
        buttonLayout.addWidget(self.pbLogin, alignment=Qt.Alignment())
        buttonLayout.addWidget(self.pbCancel, alignment=Qt.Alignment())
        layout.addLayout(buttonLayout)
        self.setLayout(layout)

    def different_main(self):
        self.leName.setReadOnly(True)
        self.plainTextEdit.setReadOnly(True)
        self.pbLogin.clicked.connect(self.login01)
        self.regist = register()
        self.leName.setText(str(self.regist.getCVolumeSerialNumber()))

    def different_encrypter(self):
        self.plainTextEdit.setEnabled(False)
        self.lePassword.setReadOnly(True)
        self.pbLogin.clicked.connect(self.login02)

    def login01(self):
        key = self.lePassword.text()
        print(key)
        print(self.regist.regist(key))
        if self.regist.regist(key):
            print("ok")
            self.accept()  # 关闭对话框并返回1
        else:
            QMessageBox.critical(self, '错误', '注册码错误')

    def login02(self):
        code = str(self.leName.text())
        enc = encrypter()
        code1 = enc.DesEncrypt(code)
        code1 = code1.decode("ascii")
        self.lePassword.setText(code1)
