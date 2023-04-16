# -*- coding: utf-8 -*-
# 导入一些模块

import os

from interface.languagecheck import languageok

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
##import linear
import xlwt
import xlrd
from xlutils.copy import copy as copy1
# from qtpy import QtWidgets, QtCore
##from languagecheck import languageok
import os.path
import win32api, win32con
import win32gui, win32print
from utils.util import *
from calculation_logic.Canvas import *
# from plt_canvas import *
# from dicom_widget_data import DicomData,DicomWidget

from scipy import stats

##import matplotlib

C_R = [43, 55, 64, 70, 80]
C_R = [53, 63, 73, 77, 83]
try:
    ##    from PyQt5 import QtCore,QtGui
    ##    import PyQt5.QtWidgets as QtGui
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QT_VERSION_STR, QSize
    from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QFont, QIcon  # ,QStringListModel

except ImportError:
    try:
        from PyQt4 import QtCore, QtGui
        from PyQt4.QtCore import QT_VERSION_STR, SIGNAL, QDir, QTranslator  # Qt, QRectF, pyqtSignal,
        from PyQt4.QtGui import QFileDialog, QDialog, QApplication, QWidget, QFont, \
            QPushButton, QLabel, QLineEdit, QHBoxLayout, QFormLayout, QStringListModel, QMessageBox, QVBoxLayout

    except ImportError:
        raise ImportError("QtImageViewer: Requires PyQt5 or PyQt4.")

##from PyQt4 import QtCore, QtGui
##from PyQt4.QtCore import SIGNAL
##from PyQt4.QtGui import QFileDialog, QDialog, QApplication, QWidget, \
##    QPushButton, QLabel, QLineEdit, QHBoxLayout, QFormLayout,QStringListModel,QMessageBox,QVBoxLayout
##from PyQt4.QtCore import QDir,QTranslator
from PyQt5 import QtCore
from calculation_logic.water import Water_Phantom
from calculation_logic.linear import *
from calculation_logic.resolution import *
from calculation_logic.thickness import SpiralBeads, CT_phantom

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except:  # AttributeError:
    def _fromUtf8(s):
        return s
try:
    _encoding = QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)


class Viewer(QMainWindow):
    def __init__(self, path=None, min1=0, max1=0, resolution=0, av=0, lcd=0, noise=0, homogeneity=0, mean1=0):
        super(Viewer, self).__init__()
        # width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        # height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)#没用到可以删掉
        ##        print(width,height)
        if (win32api.GetSystemMetrics(win32con.SM_CXSCREEN) < 1920):  # 获取系统分辨率
            scale_factor = 1.0  # 控件布局缩放比例以适应Windows设置中的缩放比例
        else:
            scale_factor = 2.0
        # 初始化一些值
        if (scale_factor == 1.0):
            self.setMinimumSize(900, 600)  # self.setMinimumSize(1350, 750)#分辨率小于1920的情况下默认窗口大小
            self.move(0, 0)
        else:
            self.setMinimumSize(1700, 900)  # (1900,1000)
            self.move(0, 0)
        self.cb = 1

        self.mgy = 0
        self.mgy1 = 0
        self.slice8_biaocheng = 0
        self.slice8_shice = 0
        self.slice8_err = 0
        self.slice28_biaocheng = 0
        self.slice28_shice = 0
        self.slice28_err = 0
        self.slice2_biaocheng = 0
        self.slice2_shice = 0
        self.slice2_err = 0
        self.place = 0
        self.setWindowTitle(_translate("", "医用CT成像设备质量检测系统", None))
        self.file = None
        self.array3 = [0, 0, 0, 0, 0, 0, 0, 0]
        self.high_hu = 2000
        self.low_hu = -1024
        self.noise = noise
        self.min1 = min1
        self.max1 = max1
        self.mean1 = mean1
        self.lcd = lcd
        self.resolution = resolution
        self.homogeneity = homogeneity
        self.av = av
        self.av2 = 0
        self.av3 = 0
        self.av5 = 0
        self.av7 = 0
        self.a0 = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 0
        self.a6 = 0
        self.a7 = 0
        self.a8 = 0
        self.a9 = 0
        self.a71 = 0
        self.a72 = 0
        self.a73 = 0
        self.a74 = 0
        self.a81 = 0
        self.a75 = 0
        self.a82 = 0
        self.a83 = 0
        self.a84 = 0
        self.a85 = 0
        self.array1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.dcmKVP = 0
        self.dcmXRayTubeCurrent = 0
        self.dcmExposureTime = 0
        self.ctLilunDef = [-1000.0, -630.0, -100.0, 120.0, 365.0, 550.0, 1000.0, 1280.0]
        self.pix_label = PhotoViewer(
            parent=self)  # PltCanvas(parent = self,dcm = None)#DicomWidget(parent = self, kwargs = None)  #pix_label为Dicom图片显示部件
        self.resolution10 = 0
        self.resolution50 = 0
        self.resolution10x = 0
        self.resolution50x = 0
        self.mid = 0
        self.length = 255
        self.diam = 166.3  # 钨珠直径
        self.pitch = 90  # 钨珠螺距
        self.beadsnum = 180  # 钨珠数量
        self.CT_err = 0
        self.pix_label.resize(512, 512)
        self.ShowPos = False
        self.KeepWindowSetting = False
        self.l = False
        self.h = False
        self.orientation = False
        self.D_Phantom = True
        scroll_area = QtWidgets.QScrollArea(self)  # 滚动窗口，包含大部分窗口部件，目前未实现滚动
        scroll_area.setWidgetResizable(True)  # 设置窗口内部件可跟随窗口大小变化

        self.widget = QWidget(scroll_area)  # wiget部分包括各指标计算部件

        # 对比度滑块设定
        self.horizontalSlider = QSlider(scroll_area)
        ##        self.horizontalSlider.setGeometry(QtCore.QRect(0, 514, 512, 19))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider1 = QSlider(scroll_area)
        ##        self.horizontalSlider1.setGeometry(QtCore.QRect(0, 542, 512, 19))
        self.horizontalSlider1.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider1.setObjectName(_fromUtf8("horizontalSlider1"))
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider1.setMinimum(0)
        self.horizontalSlider1.setMaximum(255)
        self.horizontalSlider1.setSliderPosition(self.horizontalSlider1.maximum())

        self.groupBox_duibidu = QGroupBox()  # 对比度调节组合框
        self.groupBox_duibidu.setMinimumSize(200, 100)
        self.gridLayout_duibidu = QGridLayout(self.groupBox_duibidu)
        self.groupBox_duibidu.setTitle(_translate("Viewer", "对比度调节", None))
        self.gridLayout_duibidu.addWidget(self.horizontalSlider, 0, 0)
        self.gridLayout_duibidu.addWidget(self.horizontalSlider1, 2, 0)

        self.widget.setObjectName(_fromUtf8("widget"))
        ##        self.gridLayout_all = QGridLayout(self.widget)    #为wiget设置网格布局
        ##        self.gridLayout_all.setObjectName(_fromUtf8("gridLayout_all"))
        groupBox_scroll = QGroupBox()

        self.gridLayout_scroll = QGridLayout(groupBox_scroll)  # scroll_area内添加网格布局，包括图片显示区、对比度调节区、指标计算
        self.gridLayout_scroll.addWidget(self.pix_label, 0, 0)  # 将图片显示加入gridLayout_scroll布局
        self.gridLayout_scroll.addWidget(self.widget, 0, 1)  # 将指标计算部分加入gridLayout_scroll布局
        ##        self.gridLayout_scroll.addWidget(self.groupBox_duibidu,1,0) #将对比度调节部分加入gridLayout_scroll布局
        self.gridLayout_scroll.setColumnStretch(0, 5)  # 设置布局内每一列伸展的权重
        self.gridLayout_scroll.setColumnStretch(1, 1)
        ##        self.gridLayout_scroll.setColumnStretch(2, 1)

        # 各指标计算控件设置
        # 1、机架倾角
        self.groupBox_qingjiao = QGroupBox(self.widget)
        self.gridLayout_qingjiao = QGridLayout(self.groupBox_qingjiao)
        self.lineEdit_angle = QLineEdit(self.groupBox_qingjiao)
        self.lineEdit_angle.setEnabled(0)
        self.pushButton_angle = QPushButton(self.groupBox_qingjiao)
        self.pushButton_angle.setText(_translate("", "计算倾角", None))
        self.gridLayout_qingjiao.addWidget(self.pushButton_angle, 0, 0, 1, 1)
        self.gridLayout_qingjiao.addWidget(self.lineEdit_angle, 0, 1, 1, 1)

        # 2、水模体指标计算
        self.groupBox_water = QGroupBox(self.widget)
        self.gridLayout_water = QGridLayout(self.groupBox_water)
        self.water_pushButton = QPushButton(self.groupBox_water)
        self.water_label_CT = QLabel(self.groupBox_water)
        self.water_lineEdit_CT = QLineEdit(self.groupBox_water)
        self.water_lineEdit_CT.setReadOnly(True)
        self.water_label_noise = QLabel(self.groupBox_water)
        self.water_lineEdit_noise = QLineEdit(self.groupBox_water)
        self.water_label_junyun = QLabel(self.groupBox_water)
        self.water_lineEdit_junyun = QLineEdit(self.groupBox_water)
        self.water_label_lcd = QLabel(self.groupBox_water)
        self.water_label_lcd_2 = QLabel(self.groupBox_water)
        self.water_label_lcd_3 = QLabel(self.groupBox_water)
        self.water_label_lcd_5 = QLabel(self.groupBox_water)
        self.water_label_lcd_7 = QLabel(self.groupBox_water)
        self.water_lineEdit_lcd_2 = QLineEdit(self.groupBox_water)
        self.water_lineEdit_lcd_3 = QLineEdit(self.groupBox_water)
        self.water_lineEdit_lcd_5 = QLineEdit(self.groupBox_water)
        self.water_lineEdit_lcd_7 = QLineEdit(self.groupBox_water)
        self.groupBox_water.setObjectName(_fromUtf8("groupBox_water"))
        self.gridLayout_water.setObjectName(_fromUtf8("gridLayout_water"))
        self.water_pushButton.setObjectName(_fromUtf8("water_pushButton"))
        self.water_label_CT.setObjectName(_fromUtf8("water_label_CT"))
        self.water_lineEdit_CT.setObjectName(_fromUtf8("water_lineEdit_CT"))
        self.water_label_noise.setObjectName(_fromUtf8("water_label_noise"))
        self.water_lineEdit_noise.setObjectName(_fromUtf8("water_lineEdit_noise"))
        self.water_label_junyun.setObjectName(_fromUtf8("water_label_junyun"))
        self.water_lineEdit_junyun.setObjectName(_fromUtf8("water_lineEdit_junyun"))
        self.water_label_lcd.setObjectName(_fromUtf8("water_label_lcd"))
        self.water_checkbox1 = QCheckBox(self.groupBox_water)
        self.water_checkbox2 = QCheckBox(self.groupBox_water)
        self.water_checkbox3 = QCheckBox(self.groupBox_water)
        self.water_checkbox4 = QCheckBox(self.groupBox_water)
        self.water_label_lcd_2.setObjectName(_fromUtf8("2"))
        self.water_label_lcd_3.setObjectName(_fromUtf8("3"))
        self.water_label_lcd_5.setObjectName(_fromUtf8("5"))
        self.water_label_lcd_7.setObjectName(_fromUtf8("7"))
        self.water_lineEdit_lcd_2.setObjectName(_fromUtf8("water_lineEdit_lcd_2"))
        self.water_lineEdit_lcd_3.setObjectName(_fromUtf8("water_lineEdit_lcd_3"))
        self.water_lineEdit_lcd_5.setObjectName(_fromUtf8("water_lineEdit_lcd_5"))
        self.water_lineEdit_lcd_7.setObjectName(_fromUtf8("water_lineEdit_lcd_7"))
        self.gridLayout_water.addWidget(self.water_pushButton, 0, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_CT, 1, 0, 1, 4)
        self.gridLayout_water.addWidget(self.water_lineEdit_CT, 1, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_noise, 2, 0, 1, 4)
        self.gridLayout_water.addWidget(self.water_lineEdit_noise, 2, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_junyun, 3, 0, 1, 4)
        self.gridLayout_water.addWidget(self.water_lineEdit_junyun, 3, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_label_lcd, 4, 0, 1, 4)
        vbox = QVBoxLayout()  # 纵向布局
        hbox1 = QHBoxLayout()  # 横向布局
        hbox2 = QHBoxLayout()  # 横向布局

        hbox1.addWidget(self.water_checkbox1)  # , 0, 0, 1, 1)
        hbox1.addWidget(self.water_label_lcd_2)  # , 0, 1, 1, 1)
        hbox1.addWidget(self.water_lineEdit_lcd_2)  # , 0, 2, 1, 3)

        hbox1.addWidget(self.water_checkbox2)  # , 0, 5, 1, 1)
        hbox1.addWidget(self.water_label_lcd_3)  # , 0, 6, 1, 1)
        hbox1.addWidget(self.water_lineEdit_lcd_3)  # , 0, 7, 1, 3)

        hbox2.addWidget(self.water_checkbox3)  # , 1, 0, 1, 1)
        hbox2.addWidget(self.water_label_lcd_5)  # , 1, 1, 1, 1)
        hbox2.addWidget(self.water_lineEdit_lcd_5)  # , 1, 2, 1, 3)

        hbox2.addWidget(self.water_checkbox4)  # , 1, 5, 1, 1)
        hbox2.addWidget(self.water_label_lcd_7)  # , 1, 6, 1, 1)
        hbox2.addWidget(self.water_lineEdit_lcd_7)  # , 1, 7, 1, 3)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.gridLayout_water.addLayout(vbox, 5, 0, 2, 8)
        self.water_pushButton_all = QPushButton(self.groupBox_water)
        self.water_pushButton_all.setObjectName(_fromUtf8("water_pushButton_all"))

        self.water_checkbox1.setObjectName(_fromUtf8("water_checkbox1"))
        self.water_checkbox2.setObjectName(_fromUtf8("water_checkbox2"))
        self.water_checkbox3.setObjectName(_fromUtf8("water_checkbox3"))
        self.water_checkbox4.setObjectName(_fromUtf8("water_checkbox4"))
        self.water_pushButton_jz = QPushButton(self.groupBox_water)
        self.water_pushButton_jz.setObjectName(_fromUtf8("LCD值"))
        self.gridLayout_water.addWidget(self.water_pushButton_jz, 7, 1, 1, 1)
        self.water_lineEdit_jz = QLineEdit(self.groupBox_water)
        self.water_lineEdit_jz.setObjectName(_fromUtf8("water_lineEdit_jz"))
        self.gridLayout_water.addWidget(self.water_lineEdit_jz, 7, 6, 1, 2)
        self.gridLayout_water.addWidget(self.water_pushButton_all, 8, 0, 1, 8)

        # 3、CT线性值指标计算
        self.groupBox_linear = QGroupBox(self.widget)
        self.gridLayout_linear = QGridLayout(self.groupBox_linear)
        self.linear_pushButton = QPushButton(self.groupBox_linear)
        self.linear_label_celiang = QLabel(self.groupBox_linear)
        self.linear_lineEdit_ctceliang1 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang2 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang3 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang4 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang5 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang6 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang7 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctceliang8 = QLineEdit(self.groupBox_linear)
        self.linear_pushButton_all = QPushButton(self.groupBox_linear)
        self.linear_pushButton_all.setObjectName(_fromUtf8("linear_pushButton_all"))
        self.linear_label_lilun = QLabel(self.groupBox_linear)
        self.checkbox1 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun1 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun1.setEnabled(False)
        self.checkbox2 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun2 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun2.setEnabled(False)
        self.checkbox3 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun3 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun3.setEnabled(False)
        self.checkbox4 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun4 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun4.setEnabled(False)
        self.checkbox5 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun5 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun5.setEnabled(False)
        self.checkbox6 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun6 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun6.setEnabled(False)
        self.checkbox7 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun7 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun7.setEnabled(False)
        self.checkbox8 = QCheckBox(self.groupBox_linear)
        self.linear_lineEdit_ctlilun8 = QLineEdit(self.groupBox_linear)
        self.linear_lineEdit_ctlilun8.setEnabled(False)
        self.pushButton_ctxianxing = QPushButton(self.groupBox_linear)
        self.linear_label_CT = QLabel(self.groupBox_linear)
        self.linear_lineEdit_CT = QLineEdit(self.groupBox_linear)
        self.groupBox_linear.setObjectName(_fromUtf8("groupBox_linear"))
        self.gridLayout_linear.setObjectName(_fromUtf8("gridLayout_linear"))
        self.linear_pushButton.setObjectName(_fromUtf8("linear_pushButton"))
        self.linear_label_celiang.setObjectName(_fromUtf8("linear_label_celiang"))
        self.linear_lineEdit_ctceliang1.setObjectName(_fromUtf8("linear_lineEdit_ctceliang1"))
        self.linear_lineEdit_ctceliang2.setObjectName(_fromUtf8("linear_lineEdit_ctceliang2"))
        self.linear_lineEdit_ctceliang3.setObjectName(_fromUtf8("linear_lineEdit_ctceliang3"))
        self.linear_lineEdit_ctceliang4.setObjectName(_fromUtf8("linear_lineEdit_ctceliang4"))
        self.linear_lineEdit_ctceliang5.setObjectName(_fromUtf8("linear_lineEdit_ctceliang5"))
        self.linear_lineEdit_ctceliang6.setObjectName(_fromUtf8("linear_lineEdit_ctceliang6"))
        self.linear_lineEdit_ctceliang7.setObjectName(_fromUtf8("linear_lineEdit_ctceliang7"))
        self.linear_lineEdit_ctceliang8.setObjectName(_fromUtf8("linear_lineEdit_ctceliang8"))
        self.linear_label_lilun.setObjectName(_fromUtf8("linear_label_lilun"))
        self.checkbox1.setObjectName(_fromUtf8("checkbox1"))
        self.linear_lineEdit_ctlilun1.setObjectName(_fromUtf8("linear_lineEdit_ctlilun1"))
        self.checkbox2.setObjectName(_fromUtf8("checkbox2"))
        self.linear_lineEdit_ctlilun2.setObjectName(_fromUtf8("linear_lineEdit_ctlilun2"))
        self.checkbox3.setObjectName(_fromUtf8("checkbox3"))
        self.linear_lineEdit_ctlilun3.setObjectName(_fromUtf8("linear_lineEdit_ctlilun3"))
        self.checkbox4.setObjectName(_fromUtf8("checkbox4"))
        self.linear_lineEdit_ctlilun4.setObjectName(_fromUtf8("linear_lineEdit_ctlilun4"))
        self.linear_lineEdit_ctlilun5.setObjectName(_fromUtf8("linear_lineEdit_ctlilun5"))
        self.checkbox5.setObjectName(_fromUtf8("checkbox5"))
        self.checkbox6.setObjectName(_fromUtf8("checkbox6"))
        self.linear_lineEdit_ctlilun6.setObjectName(_fromUtf8("linear_lineEdit_ctlilun6"))
        self.checkbox7.setObjectName(_fromUtf8("checkbox7"))
        self.linear_lineEdit_ctlilun7.setObjectName(_fromUtf8("linear_lineEdit_ctlilun7"))
        self.checkbox8.setObjectName(_fromUtf8("checkbox8"))
        self.linear_lineEdit_ctlilun8.setObjectName(_fromUtf8("linear_lineEdit_ctlilun8"))
        self.pushButton_ctxianxing.setObjectName(_fromUtf8("pushButton_ctxianxing"))
        self.linear_label_CT.setObjectName(_fromUtf8("linear_label_CT"))
        self.linear_lineEdit_CT.setObjectName(_fromUtf8("linear_lineEdit_CT"))
        self.gridLayout_linear.addWidget(self.linear_pushButton, 1, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_label_celiang, 1, 0, 1, 6)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang1, 2, 0, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang2, 2, 2, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang3, 2, 4, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang4, 2, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang5, 3, 0, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang6, 3, 2, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang7, 3, 4, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctceliang8, 3, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_pushButton_all, 4, 0, 1, 8)
        self.gridLayout_linear.addWidget(self.linear_label_lilun, 5, 0, 1, 6)
        self.gridLayout_linear.addWidget(self.checkbox1, 6, 0, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun1, 6, 1, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox2, 6, 2, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun2, 6, 3, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox3, 6, 4, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun3, 6, 5, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox4, 6, 6, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun4, 6, 7, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox5, 7, 0, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun5, 7, 1, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox6, 7, 2, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun6, 7, 3, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox7, 7, 4, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun7, 7, 5, 1, 1)
        self.gridLayout_linear.addWidget(self.checkbox8, 7, 6, 1, 1)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_ctlilun8, 7, 7, 1, 1)
        self.gridLayout_linear.addWidget(self.pushButton_ctxianxing, 9, 6, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_label_CT, 9, 0, 1, 2)
        self.gridLayout_linear.addWidget(self.linear_lineEdit_CT, 9, 2, 1, 2)

        self.linear_lineEdit_ctlilun1.setText("%.1f" % (self.ctLilunDef[0]))
        self.linear_lineEdit_ctlilun2.setText("%.1f" % (self.ctLilunDef[1]))
        self.linear_lineEdit_ctlilun3.setText("%.1f" % (self.ctLilunDef[2]))
        self.linear_lineEdit_ctlilun4.setText("%.1f" % (self.ctLilunDef[3]))
        self.linear_lineEdit_ctlilun5.setText("%.1f" % (self.ctLilunDef[4]))
        self.linear_lineEdit_ctlilun6.setText("%.1f" % (self.ctLilunDef[5]))
        self.linear_lineEdit_ctlilun7.setText("%.1f" % (self.ctLilunDef[6]))
        self.linear_lineEdit_ctlilun8.setText("%.1f" % (self.ctLilunDef[7]))
        # 4、层厚值指标计算
        self.groupBox_sliceThickness = QGroupBox(self.widget)
        self.gridLayout_sliceThickness = QGridLayout(self.groupBox_sliceThickness)
        self.label_cenghou8 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_8 = QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_label_bc8 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_bc8 = QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_shice8 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_shice8 = QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_error8 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_error8 = QLineEdit(self.groupBox_sliceThickness)
        self.label_cenghou28 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_28 = QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_label_bc28 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_bc28 = QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_shice28 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_shice28 = QLineEdit(self.groupBox_sliceThickness)
        self.sliceThickness_label_error28 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_lineEdit_error28 = QLineEdit(self.groupBox_sliceThickness)
        self.label_cenghou2 = QLabel(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_2 = QPushButton()
        self.sliceThickness_label_bc2 = QLabel()
        self.sliceThickness_lineEdit_bc2 = QLineEdit()
        self.sliceThickness_label_shice2 = QLabel()
        self.sliceThickness_lineEdit_shice2 = QLineEdit()
        self.sliceThickness_label_error2 = QLabel()
        self.sliceThickness_lineEdit_error2 = QLineEdit()
        self.groupBox_sliceThickness.setObjectName(_fromUtf8("groupBox_sliceThickness"))
        self.gridLayout_sliceThickness.setObjectName(_fromUtf8("gridLayout_sliceThickness"))
        self.sliceThickness_pushButton_8.setObjectName(_fromUtf8("sliceThickness_pushButton_8"))
        self.sliceThickness_label_bc8.setObjectName(_fromUtf8("sliceThickness_label_bc8"))
        self.sliceThickness_lineEdit_bc8.setObjectName(_fromUtf8("sliceThickness_lineEdit_bc8"))
        self.sliceThickness_label_shice8.setObjectName(_fromUtf8("sliceThickness_label_shice8"))
        self.sliceThickness_lineEdit_shice8.setObjectName(_fromUtf8("sliceThickness_lineEdit_shice8"))
        self.sliceThickness_label_error8.setObjectName(_fromUtf8("sliceThickness_label_error8"))
        self.sliceThickness_lineEdit_error8.setObjectName(_fromUtf8("sliceThickness_lineEdit_error8"))
        self.sliceThickness_pushButton_28.setObjectName(_fromUtf8("sliceThickness_pushButton_28"))
        self.sliceThickness_label_bc28.setObjectName(_fromUtf8("sliceThickness_label_bc28"))
        self.sliceThickness_lineEdit_bc28.setObjectName(_fromUtf8("sliceThickness_lineEdit_bc28"))
        self.sliceThickness_label_shice28.setObjectName(_fromUtf8("sliceThickness_label_shice28"))
        self.sliceThickness_lineEdit_shice28.setObjectName(_fromUtf8("sliceThickness_lineEdit_shice28"))
        self.sliceThickness_label_error28.setObjectName(_fromUtf8("sliceThickness_label_error28"))
        self.sliceThickness_lineEdit_error28.setObjectName(_fromUtf8("sliceThickness_lineEdit_error28"))
        self.sliceThickness_pushButton_2.setObjectName(_fromUtf8("sliceThickness_pushButton_2"))
        self.sliceThickness_label_bc2.setObjectName(_fromUtf8("sliceThickness_label_bc2"))
        self.sliceThickness_lineEdit_bc2.setObjectName(_fromUtf8("sliceThickness_lineEdit_bc2"))
        self.sliceThickness_label_shice2.setObjectName(_fromUtf8("sliceThickness_label_shice2"))
        self.sliceThickness_lineEdit_shice2.setObjectName(_fromUtf8("sliceThickness_lineEdit_shice2"))
        self.sliceThickness_label_error2.setObjectName(_fromUtf8("sliceThickness_label_error2"))
        self.sliceThickness_lineEdit_error2.setObjectName(_fromUtf8("sliceThickness_lineEdit_error2"))

        self.sliceThickness_pushButton_all = QPushButton(self.groupBox_sliceThickness)
        self.sliceThickness_pushButton_all.setObjectName(_fromUtf8("sliceThickness_pushButton_all"))
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_8, 0, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_bc8, 1, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_bc8, 1, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_shice8, 2, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_shice8, 2, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_error8, 3, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_error8, 3, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_28, 4, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_bc28, 5, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_bc28, 5, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_shice28, 6, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_shice28, 6, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_error28, 7, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_error28, 7, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_2, 8, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_bc2, 9, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_bc2, 9, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_shice2, 10, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_shice2, 10, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_label_error2, 11, 0, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_lineEdit_error2, 11, 2, 1, 1)
        self.gridLayout_sliceThickness.addWidget(self.sliceThickness_pushButton_all, 12, 0, 1, 4)
        # 5、空间分辨率指标计算
        self.groupBox_resolution = QGroupBox(self.widget)
        self.gridLayout_resolution = QGridLayout(self.groupBox_resolution)
        self.resolution_label_x = QLabel(self.groupBox_resolution)
        self.resolution_label_changgui = QLabel(self.groupBox_resolution)
        self.resolution_label_gaofenbian = QLabel(self.groupBox_resolution)
        self.resolution_label_y = QLabel(self.groupBox_resolution)
        self.resolution_lineEdit_y = QLineEdit(self.groupBox_resolution)
        self.resolution_lineEdit_x = QLineEdit(self.groupBox_resolution)
        self.resolution_pushButton_1 = QPushButton(self.groupBox_resolution)
        self.resolution_pushButton_2 = QPushButton(self.groupBox_resolution)

        self.resolution_checkbox1 = QCheckBox(self.groupBox_resolution)
        self.resolution_label_show1 = QLabel(self.groupBox_resolution)
        self.resolution_checkbox2 = QCheckBox(self.groupBox_resolution)
        self.resolution_label_show2 = QLabel(self.groupBox_resolution)

        self.resolution_lineEdit_C_10 = QLineEdit(self.groupBox_resolution)
        self.resolution_lineEdit_G_10 = QLineEdit(self.groupBox_resolution)
        self.resolution_label_C_10 = QLabel(self.groupBox_resolution)
        self.resolution_label_G_10 = QLabel(self.groupBox_resolution)
        self.resolution_label_G_50 = QLabel(self.groupBox_resolution)
        self.resolution_label_C_50 = QLabel(self.groupBox_resolution)
        self.resolution_lineEdit_C_50 = QLineEdit(self.groupBox_resolution)
        self.resolution_lineEdit_G_50 = QLineEdit(self.groupBox_resolution)
        self.groupBox_resolution.setObjectName(_fromUtf8("groupBox_resolution"))
        self.gridLayout_resolution.setObjectName(_fromUtf8("gridLayout_resolution"))
        self.resolution_label_x.setObjectName(_fromUtf8("resolution_label_x"))
        self.resolution_label_changgui.setObjectName(_fromUtf8("resolution_label_changgui"))
        self.resolution_label_gaofenbian.setObjectName(_fromUtf8("resolution_label_gaofenbian"))
        self.resolution_label_y.setObjectName(_fromUtf8("resolution_label_y"))
        self.resolution_lineEdit_y.setObjectName(_fromUtf8("resolution_lineEdit_y"))
        self.resolution_lineEdit_x.setObjectName(_fromUtf8("resolution_lineEdit_x"))
        self.resolution_pushButton_1.setObjectName(_fromUtf8("resolution_pushButton_1"))
        self.resolution_pushButton_2.setObjectName(_fromUtf8("resolution_pushButton_2"))
        self.resolution_lineEdit_C_10.setObjectName(_fromUtf8("resolution_lineEdit_C_10"))
        self.resolution_lineEdit_G_10.setObjectName(_fromUtf8("resolution_lineEdit_G_10"))
        self.resolution_label_C_10.setObjectName(_fromUtf8("resolution_label_C_10"))
        self.resolution_label_G_10.setObjectName(_fromUtf8("resolution_label_G_10"))
        self.resolution_label_G_50.setObjectName(_fromUtf8("resolution_label_G_50"))
        self.resolution_label_C_50.setObjectName(_fromUtf8("resolution_label_C_50"))
        self.resolution_lineEdit_C_50.setObjectName(_fromUtf8("resolution_lineEdit_C_50"))
        self.resolution_lineEdit_G_50.setObjectName(_fromUtf8("resolution_lineEdit_G_50"))
        self.resolution_pushButton_all = QPushButton(self.groupBox_resolution)
        self.resolution_pushButton_all.setObjectName(_fromUtf8("resolution_pushButton_all"))
        self.gridLayout_resolution.addWidget(self.resolution_label_x, 0, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_changgui, 1, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_gaofenbian, 1, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_y, 0, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_y, 0, 3, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_x, 0, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_pushButton_1, 1, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_pushButton_2, 1, 3, 1, 1)

        self.gridLayout_resolution.addWidget(self.resolution_checkbox1, 2, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_show1, 2, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_checkbox2, 2, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_show2, 2, 3, 1, 1)

        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_C_10, 3, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_G_10, 3, 3, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_C_10, 3, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_G_10, 3, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_G_50, 4, 2, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_label_C_50, 4, 0, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_C_50, 4, 1, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_lineEdit_G_50, 4, 3, 1, 1)
        self.gridLayout_resolution.addWidget(self.resolution_pushButton_all, 5, 0, 1, 4)

        # 6、剂量指数
        self.groupBox_5 = QGroupBox(self.widget)
        self.gridLayout_7 = QGridLayout(self.groupBox_5)
        self.pushButton_head = QPushButton(self.groupBox_5)
        self.label_head_ceter = QLabel(self.groupBox_5)
        self.label_head = QLabel(self.groupBox_5)
        self.label_body = QLabel(self.groupBox_5)
        self.label_head_0 = QLabel(self.groupBox_5)
        self.label_head_3 = QLabel(self.groupBox_5)
        self.label_head_6 = QLabel(self.groupBox_5)
        self.label_head_9 = QLabel(self.groupBox_5)
        self.label_head_CTDIw = QLabel(self.groupBox_5)
        self.lineEdit_head_center = QLineEdit(self.groupBox_5)
        self.lineEdit_head_0 = QLineEdit(self.groupBox_5)
        self.lineEdit_head_3 = QLineEdit(self.groupBox_5)
        self.lineEdit_head_6 = QLineEdit(self.groupBox_5)
        self.lineEdit_head_9 = QLineEdit(self.groupBox_5)
        self.lineEdit_head_CTDIw = QLineEdit(self.groupBox_5)
        self.pushButton_body = QPushButton(self.groupBox_5)
        self.label_body_ceter = QLabel(self.groupBox_5)
        self.label_body_0 = QLabel(self.groupBox_5)
        self.label_body_3 = QLabel(self.groupBox_5)
        self.label_body_6 = QLabel(self.groupBox_5)
        self.label_body_9 = QLabel(self.groupBox_5)
        self.label_body_CTDIw = QLabel(self.groupBox_5)
        self.lineEdit_body_center = QLineEdit(self.groupBox_5)
        self.lineEdit_body_0 = QLineEdit(self.groupBox_5)
        self.lineEdit_body_3 = QLineEdit(self.groupBox_5)
        self.lineEdit_body_6 = QLineEdit(self.groupBox_5)
        self.lineEdit_body_9 = QLineEdit(self.groupBox_5)
        self.lineEdit_body_CTDIw = QLineEdit(self.groupBox_5)
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.pushButton_head.setObjectName(_fromUtf8("pushButton_head"))
        self.label_head_ceter.setObjectName(_fromUtf8("label_head_ceter"))
        self.label_head.setObjectName(_fromUtf8("label_head"))
        self.label_body.setObjectName(_fromUtf8("label_body"))
        self.label_head_0.setObjectName(_fromUtf8("label_head_0"))
        self.label_head_3.setObjectName(_fromUtf8("label_head_3"))
        self.label_head_6.setObjectName(_fromUtf8("label_head_6"))
        self.label_head_9.setObjectName(_fromUtf8("label_head_9"))
        self.label_head_CTDIw.setObjectName(_fromUtf8("label_head_CTDIw"))
        self.lineEdit_head_center.setObjectName(_fromUtf8("lineEdit_head_center"))
        self.lineEdit_head_0.setObjectName(_fromUtf8("lineEdit_head_0"))
        self.lineEdit_head_3.setObjectName(_fromUtf8("lineEdit_head_3"))
        self.lineEdit_head_6.setObjectName(_fromUtf8("lineEdit_head_6"))
        self.lineEdit_head_9.setObjectName(_fromUtf8("lineEdit_head_9"))
        self.lineEdit_head_CTDIw.setObjectName(_fromUtf8("lineEdit_head_CTDIw"))
        self.pushButton_body.setObjectName(_fromUtf8("pushButton_body"))
        self.label_body_ceter.setObjectName(_fromUtf8("label_body_ceter"))
        self.label_body_0.setObjectName(_fromUtf8("label_body_0"))
        self.label_body_3.setObjectName(_fromUtf8("label_body_3"))
        self.label_body_6.setObjectName(_fromUtf8("label_body_6"))
        self.label_body_9.setObjectName(_fromUtf8("label_body_9"))
        self.label_body_CTDIw.setObjectName(_fromUtf8("label_body_CTDIw"))
        self.lineEdit_body_center.setObjectName(_fromUtf8("lineEdit_body_center"))
        self.lineEdit_body_0.setObjectName(_fromUtf8("lineEdit_body_0"))
        self.lineEdit_body_3.setObjectName(_fromUtf8("lineEdit_body_3"))
        self.lineEdit_body_6.setObjectName(_fromUtf8("lineEdit_body_6"))
        self.lineEdit_body_9.setObjectName(_fromUtf8("lineEdit_body_9"))
        self.lineEdit_body_CTDIw.setObjectName(_fromUtf8("lineEdit_head_CTDIw"))
        self.gridLayout_7.addWidget(self.pushButton_head, 0, 1, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_ceter, 1, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_body, 0, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_0, 2, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_3, 3, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_6, 4, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_9, 5, 0, 1, 1)
        self.gridLayout_7.addWidget(self.label_head_CTDIw, 6, 0, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_center, 1, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_0, 2, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_3, 3, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_6, 4, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_9, 5, 1, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_head_CTDIw, 6, 1, 1, 1)
        self.gridLayout_7.addWidget(self.pushButton_body, 0, 3, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_ceter, 1, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_0, 2, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_3, 3, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_6, 4, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_9, 5, 2, 1, 1)
        self.gridLayout_7.addWidget(self.label_body_CTDIw, 6, 2, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_center, 1, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_0, 2, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_3, 3, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_6, 4, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_9, 5, 3, 1, 1)
        self.gridLayout_7.addWidget(self.lineEdit_body_CTDIw, 6, 3, 1, 1)

        # 7、窗宽窗位
        self.groupBox_win = QGroupBox()
        self.gridLayout_win = QGridLayout(self.groupBox_win)
        self.lineEdit_wc = QLineEdit()
        self.lineEdit_wc.setEnabled(0)
        self.lineEdit_ww = QLineEdit()
        self.lineEdit_ww.setEnabled(0)
        self.label_wc = QLabel(self.groupBox_win)
        self.label_wc.setText(_translate("", "窗位", None))
        self.label_ww = QLabel(self.groupBox_win)
        self.label_ww.setText(_translate("", "窗宽", None))

        self.speedSlider = QSlider()
        self.speedSlider.setOrientation(QtCore.Qt.Horizontal)
        ##        self.horizontalSlider1.setGeometry(QtCore.QRect(0, 542, 512, 19))
        self.speedSlider.setMinimum(0)
        self.speedSlider.setMaximum(10)
        self.speedSlider.setSliderPosition(5)
        self.winOk = QPushButton("确定")

        self.wCheckbox = QCheckBox()
        self.wLabel = QLabel()
        self.wLabel.setText(_translate("", "显示检测标记", None))

        self.wwcCheckbox = QCheckBox()
        self.wwcLabel = QLabel()
        self.wwcLabel.setText(_translate("", "沿用此设置", None))
        self.gridLayout_qingjiao.addWidget(self.label_wc, 2, 0, 1, 1)
        self.gridLayout_qingjiao.addWidget(self.label_ww, 3, 0, 1, 1)
        self.gridLayout_qingjiao.addWidget(self.lineEdit_wc, 2, 1, 1, 1)
        self.gridLayout_qingjiao.addWidget(self.lineEdit_ww, 3, 1, 1, 1)
        self.gridLayout_qingjiao.addWidget(self.speedSlider, 4, 0, 1, 2)
        ##        self.gridLayout_qingjiao.addWidget(self.qingjiaoOk,3,1,1,1)
        self.gridLayout_qingjiao.addWidget(self.wwcCheckbox, 5, 0, 1, 1)
        self.gridLayout_qingjiao.addWidget(self.wwcLabel, 5, 1, 1, 1)

        vbox01 = QVBoxLayout()  # 纵向布局
        vbox02 = QVBoxLayout()  # 纵向布局
        hbox00 = QHBoxLayout(self.widget)  # 横向布局
        vbox01.addWidget(self.groupBox_water)
        vbox01.addWidget(self.groupBox_sliceThickness)
        vbox01.addWidget(self.groupBox_qingjiao)
        ##        vbox01.addWidget(self.groupBox_win)
        vbox02.addWidget(self.groupBox_linear)
        vbox02.addWidget(self.groupBox_resolution)
        vbox02.addWidget(self.groupBox_5)
        hbox00.addLayout(vbox01)
        hbox00.addLayout(vbox02)
        ##        testlayout = QGridLayout()

        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setWidget(groupBox_scroll)  # setLayout(hbox00)#Widget(self.widget)
        self.widget.setMinimumSize(int(self.widget.minimumSizeHint().width() * 1.3),
                                   self.widget.minimumSizeHint().height())  # *1.2
        self.widget.setMaximumHeight(int(self.widget.minimumSizeHint().height() * 1.1))

        self.setCentralWidget(scroll_area)
        self.file_dock = QtWidgets.QDockWidget("Files", self)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.file_dock)  # 左侧文件停靠
        ##        self.file_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)##
        self.file_list = QtWidgets.QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_item_change)
        self.file_dock.setWidget(self.file_list)
        self.file_dock.setMaximumWidth(150)
        self.hu_label = QtWidgets.QLabel("No image")
        self.c_label = QtWidgets.QLabel("")
        self.cw_label = QtWidgets.QLabel("")
        self.x_label = QtWidgets.QLabel("")
        self.y_label = QtWidgets.QLabel("")
        self.z_label = QtWidgets.QLabel("")
        self.use_fractional_coordinates = True
        self.ij_label = QtWidgets.QLabel("")
        self._zoom_level = 1
        self.mouse_x = -1
        self.mouse_y = -1
        self.water_pushButton_jz.setText(_translate("groupBox_water", "LCD值", None))
        self.resolution_label_changgui.setText(_translate("groupBox_sliceThickness", "常规算法", None))
        self.resolution_label_gaofenbian.setText(_translate("groupBox_sliceThickness", "高分辨算法", None))
        self.resolution_label_y.setText(_translate("groupBox_sliceThickness", "钨珠中心(y)", None))
        self.resolution_pushButton_2.setText(_translate("groupBox_sliceThickness", "计算", None))
        self.resolution_label_G_50.setText(_translate("groupBox_sliceThickness", "50%", None))
        self.resolution_label_C_50.setText(_translate("groupBox_sliceThickness", "50%", None))
        self.label_head.setText(_translate("groupBox_5", "头部模体", None))
        self.label_body.setText(_translate("groupBox_5", "体部模体", None))
        self.groupBox_water.setTitle(_translate("Viewer", "水模体指标", None))
        self.water_pushButton.setText(_translate("Viewer", "计算", None))
        self.water_label_CT.setText(_translate("Viewer", "CT值(水)", None))
        self.water_label_noise.setText(_translate("Viewer", "噪声(%)", None))
        self.water_label_junyun.setText(_translate("Viewer", "均匀性(HU)", None))
        self.water_label_lcd.setText(_translate("Viewer", "低对比可探测能力", None))
        self.water_label_lcd_2.setText(_translate("Viewer", "2", None))
        self.water_label_lcd_3.setText(_translate("Viewer", "3", None))
        self.water_label_lcd_5.setText(_translate("Viewer", "5", None))
        self.water_label_lcd_7.setText(_translate("Viewer", "7", None))
        self.water_pushButton_all.setText(_translate("Viewer", "统计列表中所有水模体指标", None))
        self.groupBox_linear.setTitle(_translate("Viewer", "CT线性值", None))
        self.linear_pushButton.setText(_translate("Viewer", "计算", None))
        self.linear_pushButton_all.setText(_translate("Viewer", "统计列表中所有测量CT值", None))
        self.linear_label_celiang.setText(_translate("Viewer", "八种材料的测量CT值", None))
        self.linear_label_lilun.setText(_translate("Viewer", "八种材料的理论CT值(用户输入)", None))
        self.pushButton_ctxianxing.setText(_translate("Viewer", "计算CT线性值", None))
        self.linear_label_CT.setText(_translate("Viewer", "CT线性值", None))
        self.groupBox_sliceThickness.setTitle(_translate("Viewer", "层厚偏差(%)", None))
        self.sliceThickness_pushButton_8.setText(_translate("Viewer", "计算", None))
        self.sliceThickness_pushButton_all.setText(_translate("Viewer", "统计列表中所有层厚", None))
        self.sliceThickness_label_bc8.setText(_translate("Viewer", "层厚标称", None))
        self.sliceThickness_label_shice8.setText(_translate("Viewer", "层厚实测值", None))
        self.sliceThickness_label_error8.setText(_translate("Viewer", "相对误差(%)", None))
        self.sliceThickness_pushButton_28.setText(_translate("Viewer", "计算", None))
        self.sliceThickness_label_bc28.setText(_translate("Viewer", "层厚标称", None))
        self.sliceThickness_label_shice28.setText(_translate("Viewer", "层厚实测值", None))
        self.sliceThickness_label_error28.setText(_translate("Viewer", "相对误差(%)", None))
        self.sliceThickness_pushButton_2.setText(_translate("Viewer", "计算", None))
        self.sliceThickness_label_bc2.setText(_translate("Viewer", "层厚标称", None))
        self.sliceThickness_label_shice2.setText(_translate("Viewer", "层厚实测值", None))
        self.sliceThickness_label_error2.setText(_translate("Viewer", "相对误差(%)", None))
        self.groupBox_resolution.setTitle(_translate("Viewer", "高对比分辨力(LP/cm)", None))
        self.resolution_label_x.setText(_translate("Viewer", "钨珠中心(x)", None))
        self.resolution_pushButton_1.setText(_translate("Viewer", "计算", None))
        self.resolution_label_C_10.setText(_translate("Viewer", "10%", None))
        self.resolution_label_G_10.setText(_translate("Viewer", "10%", None))
        self.resolution_label_show1.setText(_translate("Viewer", "显示MTF曲线", None))
        self.resolution_label_show2.setText(_translate("Viewer", "显示MTF曲线", None))
        self.resolution_pushButton_all.setText(_translate("Viewer", "统计列表中所有高对比分辨力值", None))
        self.label_head_ceter.setText(_translate("Viewer", "中心", None))
        self.label_head_CTDIw.setText(_translate("Viewer", "CTDIw", None))
        self.label_head_0.setText(_translate("Viewer", "0", None))
        self.label_head_3.setText(_translate("Viewer", "3", None))
        self.label_head_6.setText(_translate("Viewer", "6", None))
        self.label_head_9.setText(_translate("Viewer", "9", None))
        self.pushButton_head.setText(_translate("Viewer", "计算", None))
        self.lineEdit_head_center.setPlaceholderText('(mGy)')
        self.lineEdit_head_CTDIw.setPlaceholderText('(mGy)')
        self.lineEdit_head_0.setPlaceholderText('(mGy)')
        self.lineEdit_head_3.setPlaceholderText('(mGy)')
        self.lineEdit_head_6.setPlaceholderText('(mGy)')
        self.lineEdit_head_9.setPlaceholderText('(mGy)')
        self.label_body_ceter.setText(_translate("Viewer", "中心", None))
        self.label_body_CTDIw.setText(_translate("Viewer", "CTDIw", None))
        self.label_body_0.setText(_translate("Viewer", "0", None))
        self.label_body_3.setText(_translate("Viewer", "3", None))
        self.label_body_6.setText(_translate("Viewer", "6", None))
        self.label_body_9.setText(_translate("Viewer", "9", None))
        self.pushButton_body.setText(_translate("Viewer", "计算", None))
        self.lineEdit_body_center.setPlaceholderText('(mGy)')
        self.lineEdit_body_CTDIw.setPlaceholderText('(mGy)')
        self.lineEdit_body_0.setPlaceholderText('(mGy)')
        self.lineEdit_body_3.setPlaceholderText('(mGy)')
        self.lineEdit_body_6.setPlaceholderText('(mGy)')
        self.lineEdit_body_9.setPlaceholderText('(mGy)')
        self.statusBar().addPermanentWidget(self.cw_label)
        self.statusBar().addPermanentWidget(self.ij_label)
        self.statusBar().addPermanentWidget(self.x_label)
        self.statusBar().addPermanentWidget(self.y_label)
        self.statusBar().addPermanentWidget(self.z_label)
        self.statusBar().addPermanentWidget(self.hu_label)
        self.water_lineEdit_CT.setEnabled(0)
        self.water_lineEdit_noise.setEnabled(0)
        self.water_lineEdit_junyun.setEnabled(0)
        self.water_lineEdit_jz.setEnabled(0)
        # self.water_lineEdit_g1.setEnabled(0)
        self.sliceThickness_lineEdit_bc8.setEnabled(0)
        self.sliceThickness_lineEdit_shice8.setEnabled(0)
        self.sliceThickness_lineEdit_error8.setEnabled(0)
        self.linear_lineEdit_ctceliang1.setEnabled(0)
        self.linear_lineEdit_ctceliang2.setEnabled(0)
        self.linear_lineEdit_ctceliang3.setEnabled(0)
        self.linear_lineEdit_ctceliang4.setEnabled(0)
        self.linear_lineEdit_ctceliang8.setEnabled(0)
        self.linear_lineEdit_ctceliang5.setEnabled(0)
        self.linear_lineEdit_ctceliang6.setEnabled(0)
        self.linear_lineEdit_ctceliang7.setEnabled(0)
        self.lineEdit_wc.setEnabled(1)
        self.lineEdit_ww.setEnabled(1)
        self.resolution_lineEdit_C_10.setEnabled(0)
        self.linear_lineEdit_CT.setEnabled(0)
        self.water_lineEdit_lcd_2.setEnabled(0)
        self.water_lineEdit_lcd_3.setEnabled(0)
        self.sliceThickness_lineEdit_shice28.setEnabled(0)
        self.sliceThickness_lineEdit_bc28.setEnabled(0)
        self.water_lineEdit_lcd_5.setEnabled(0)
        self.water_lineEdit_lcd_7.setEnabled(0)
        self.resolution_lineEdit_G_10.setEnabled(0)
        self.sliceThickness_lineEdit_error28.setEnabled(0)
        self.sliceThickness_lineEdit_bc2.setEnabled(0)
        self.sliceThickness_lineEdit_shice2.setEnabled(0)
        self.sliceThickness_lineEdit_error2.setEnabled(0)
        self.water_label_lcd_3.setAlignment(QtCore.Qt.AlignRight)
        self.water_label_lcd_7.setAlignment(QtCore.Qt.AlignRight)
        self.water_label_lcd_5.setAlignment(QtCore.Qt.AlignRight)
        self.water_label_lcd_2.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_G_10.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_C_10.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_G_50.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_label_C_50.setAlignment(QtCore.Qt.AlignRight)
        self.resolution_lineEdit_C_50.setEnabled(0)
        self.resolution_lineEdit_G_50.setEnabled(0)
        self.resolution_label_y.setAlignment(QtCore.Qt.AlignRight)
        self.angle = 0
        self.water_pushButton.clicked.connect(self.button_click_water)
        self.water_pushButton_jz.clicked.connect(self.button_click_water_jz)
        self.linear_pushButton.clicked.connect(self.button_click_linear)
        self.sliceThickness_pushButton_8.clicked.connect(self.button_click_slice8)
        self.sliceThickness_pushButton_28.clicked.connect(self.button_click_slice28)
        self.sliceThickness_pushButton_2.clicked.connect(self.button_click_slice2)
        self.water_pushButton_all.clicked.connect(self.progress_wa)
        self.sliceThickness_pushButton_all.clicked.connect(self.progress_th)
        self.linear_pushButton_all.clicked.connect(self.progress_li)
        self.resolution_pushButton_all.clicked.connect(self.progress_MTF)
        self.pushButton_ctxianxing.clicked.connect(self.button_click_ctxianxing)
        self.linear_lineEdit_ctlilun1.textChanged.connect(self.onChanged0)
        self.linear_lineEdit_ctlilun2.textChanged.connect(self.onChanged1)
        self.linear_lineEdit_ctlilun3.textChanged.connect(self.onChanged2)
        self.linear_lineEdit_ctlilun4.textChanged.connect(self.onChanged3)
        self.linear_lineEdit_ctlilun5.textChanged.connect(self.onChanged4)
        self.linear_lineEdit_ctlilun6.textChanged.connect(self.onChanged5)
        self.linear_lineEdit_ctlilun7.textChanged.connect(self.onChanged6)
        self.linear_lineEdit_ctlilun8.textChanged.connect(self.onChanged7)
        self.resolution_lineEdit_x.textChanged.connect(self.onChanged8)
        self.resolution_lineEdit_y.textChanged.connect(self.onChanged9)
        self.lineEdit_head_center.textChanged.connect(self.onChanged11)
        self.lineEdit_head_0.textChanged.connect(self.onChanged12)
        self.lineEdit_head_3.textChanged.connect(self.onChanged13)
        self.lineEdit_head_6.textChanged.connect(self.onChanged14)
        self.lineEdit_head_9.textChanged.connect(self.onChanged15)
        self.lineEdit_body_center.textChanged.connect(self.onChanged17)
        self.lineEdit_body_0.textChanged.connect(self.onChanged18)
        self.lineEdit_body_3.textChanged.connect(self.onChanged19)
        self.lineEdit_body_6.textChanged.connect(self.onChanged20)
        self.lineEdit_body_9.textChanged.connect(self.onChanged21)
        self.resolution_pushButton_1.clicked.connect(self.button_click_resolution_1)
        self.pushButton_head.clicked.connect(self.button_click8)
        self.pushButton_body.clicked.connect(self.button_click11)
        self.resolution_pushButton_2.clicked.connect(self.button_click_resolution_2)
        self.pushButton_angle.clicked.connect(self.button_click_angle)
        self.horizontalSlider.sliderMoved.connect(self.update_mid)
        self.horizontalSlider1.sliderMoved.connect(self.update_length)
        self.horizontalSlider.valueChanged.connect(self.update_mid)
        self.horizontalSlider1.valueChanged.connect(self.update_length)

        self.checkbox1.clicked.connect(self.toggleckbox1)
        self.checkbox2.clicked.connect(self.toggleckbox2)
        self.checkbox3.clicked.connect(self.toggleckbox3)
        self.checkbox4.clicked.connect(self.toggleckbox4)
        self.checkbox5.clicked.connect(self.toggleckbox5)
        self.checkbox6.clicked.connect(self.toggleckbox6)
        self.checkbox7.clicked.connect(self.toggleckbox7)
        self.checkbox8.clicked.connect(self.toggleckbox8)
        self.wCheckbox.clicked.connect(self.wChecked)
        self.wwcCheckbox.clicked.connect(self.wwcChecked)
        self.lineEdit_wc.returnPressed.connect(self.winchange)
        self.lineEdit_ww.returnPressed.connect(self.winchange)
        self.winOk.clicked.connect(self.winchange)
        self.data = np.ndarray((512, 512), np.int8)
        self.update_cw()
        if os.path.isfile(path):
            self.load_files([path])
        elif os.path.isdir(path):
            self.load_files(dicom_files_in_dir(path))
        self.build_menu()

    def winchange(self):
        self.pix_label.l = int(self.lineEdit_wc.text()) - int(self.lineEdit_ww.text()) / 2
        self.pix_label.h = int(self.lineEdit_wc.text()) + int(self.lineEdit_ww.text()) / 2
        self.pix_label.genImage(0, 0)

    def wwcChecked(self):
        if (self.wwcCheckbox.isChecked()):
            self.KeepWindowSetting = True
            self.l = self.pix_label.l
            self.h = self.pix_label.h
        else:
            self.KeepWindowSetting = False

    def wChecked(self):
        if (self.wCheckbox.isChecked()):
            self.ShowPos = True
        else:
            self.ShowPos = False
            self.pix_label.loadLabels(xpos=[], ypos=[])
            self.pix_label.fitInView1()

    def toggleckbox1(self):
        if (self.checkbox1.isChecked()):
            self.linear_lineEdit_ctlilun1.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun1.setEnabled(False)

    def toggleckbox2(self):
        if (self.checkbox2.isChecked()):
            self.linear_lineEdit_ctlilun2.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun2.setEnabled(False)

    def toggleckbox3(self):
        if (self.checkbox3.isChecked()):
            self.linear_lineEdit_ctlilun3.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun3.setEnabled(False)

    def toggleckbox4(self):
        if (self.checkbox4.isChecked()):
            self.linear_lineEdit_ctlilun4.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun4.setEnabled(False)

    def toggleckbox5(self):
        if (self.checkbox5.isChecked()):
            self.linear_lineEdit_ctlilun5.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun5.setEnabled(False)

    def toggleckbox6(self):
        if (self.checkbox6.isChecked()):
            self.linear_lineEdit_ctlilun6.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun6.setEnabled(False)

    def toggleckbox7(self):
        if (self.checkbox7.isChecked()):
            self.linear_lineEdit_ctlilun7.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun7.setEnabled(False)

    def toggleckbox8(self):
        if (self.checkbox8.isChecked()):
            self.linear_lineEdit_ctlilun8.setEnabled(True)
        else:
            self.linear_lineEdit_ctlilun8.setEnabled(False)

    def update_mid(self):  # 函数没用
        print("update_mid")
        return
        self.mid = self.horizontalSlider.value()
        if self.mid < self.length:
            self.pix_label.update_image1()

    def update_length(self):  # 函数没用
        print("update_length")
        return
        self.length = self.horizontalSlider1.value()
        if self.mid < self.length:
            self.pix_label.update_image1()

    def onChanged0(self):
        self.a0 = float(self.linear_lineEdit_ctlilun1.text())
        self.array3[0] = 1

    def onChanged1(self):
        self.a1 = float(self.linear_lineEdit_ctlilun2.text())
        self.array3[1] = 1

    def onChanged2(self):
        self.a2 = float(self.linear_lineEdit_ctlilun3.text())
        self.array3[2] = 1

    def onChanged3(self):
        self.a3 = float(self.linear_lineEdit_ctlilun4.text())
        self.array3[3] = 1

    def onChanged4(self):
        self.a4 = float(self.linear_lineEdit_ctlilun5.text())
        self.array3[4] = 1

    def onChanged5(self):
        self.a5 = float(self.linear_lineEdit_ctlilun6.text())
        self.array3[5] = 1

    def onChanged6(self):
        self.a6 = float(self.linear_lineEdit_ctlilun7.text())
        self.array3[6] = 1

    def onChanged7(self):
        self.a7 = float(self.linear_lineEdit_ctlilun8.text())
        self.array3[7] = 1

    def onChanged8(self):
        if len(self.resolution_lineEdit_x.text()) > 2:
            print(len(self.resolution_lineEdit_x.text()))
            self.a8 = int(self.resolution_lineEdit_x.text())
            print('a8此刻输入的内容是：%s' % self.a8)
        else:
            self.a8 = 0

    def onChanged9(self):
        if len(self.resolution_lineEdit_y.text()) > 2:
            self.a9 = int(self.resolution_lineEdit_y.text())
            print('a9此刻输入的内容是：%s' % self.a9)
        else:
            self.a9 = 0

    def onChanged11(self):
        try:
            self.a71 = float(self.lineEdit_head_center.text())
            print('a71此刻输入的内容是：%s' % self.a71)
        except:
            # ("非整数")
            pass

    def onChanged12(self):
        try:
            self.a72 = float(self.lineEdit_head_0.text())
            print('a72此刻输入的内容是：%s' % self.a72)
        except:
            # ("非整数")
            pass

    def onChanged13(self):
        try:
            self.a73 = float(self.lineEdit_head_3.text())
            print('a73此刻输入的内容是：%s' % self.a73)
        except:
            # ("非整数")
            pass

    def onChanged14(self):
        try:
            self.a74 = float(self.lineEdit_head_6.text())
            print('a74此刻输入的内容是：%s' % self.a74)
        except:
            # ("非整数")
            pass

    def onChanged15(self):
        try:
            self.a75 = float(self.lineEdit_head_9.text())
            print('a75此刻输入的内容是：%s' % self.a75)
        except:
            # ("非整数")
            pass

    def onChanged17(self):
        try:
            self.a81 = float(self.lineEdit_body_center.text())
            print('a81此刻输入的内容是：%s' % self.a81)
        except:
            # ("非整数")
            pass

    def onChanged18(self):
        try:
            self.a82 = float(self.lineEdit_body_0.text())
            print('a82此刻输入的内容是：%s' % self.a82)
        except:
            # ("非整数")
            pass

    def onChanged19(self):
        try:
            self.a83 = float(self.lineEdit_body_3.text())
            print('a83此刻输入的内容是：%s' % self.a83)
        except:
            # ("非整数")
            pass

    def onChanged20(self):
        try:
            self.a84 = float(self.lineEdit_body_6.text())
            print('a84此刻输入的内容是：%s' % self.a84)
        except:
            # ("非整数")
            pass

    def onChanged21(self):
        try:
            self.a85 = float(self.lineEdit_body_9.text())
            print('a85此刻输入的内容是：%s' % self.a85)
        except:
            # ("非整数")
            pass

    def english(self):
        lan = languageok()
        lan.langu(1)
        reply = QMessageBox.information(self,
                                        '提示',
                                        '您确定使用英文界面吗？',
                                        QMessageBox.Yes | QMessageBox.No)

    def chinese(self):
        lan = languageok()
        lan.langu(0)
        reply = QMessageBox.information(self,
                                        '!',
                                        'Quit and use Chinese？',
                                        QMessageBox.Yes | QMessageBox.No)

    def button_click_angle(self):
        fname = self.file_name
        if (not self.C_Phantom) or (self.orientation != 2):
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体或非横截面图像！',
                                    QMessageBox.Ok)
            return
        if self.ShowPos == False:
            self.angle = estimate_tilt_angle(fname, IsFile=True)
            self.lineEdit_angle.setText(self.strlen(self.angle))
        else:
            try:
                self.angle, [xpos, ypos] = estimate_tilt_angle(fname, IsFile=True, ShowLabel=True)
                self.lineEdit_angle.setText(self.strlen(self.angle))
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()
            except:
                print("error in load LABELS!")

    def button_click_water(self):
        fname = self.file_name
        fname = fname.replace('/', '\\\\')
        ds = dicom.read_file(fname)
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        try:
            print("self.r = " + str(self.r))
            test = Water_Phantom(fname, r=self.r)
            if self.orientation == 2:
                av, noise = test.water_roi()
                homogeneity = test.homogeneity()
                sz1 = 2
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a2 = test.calculate_lcd(sz)
                sz1 = 3
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a3 = test.calculate_lcd(sz)
                sz1 = 5
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a5 = test.calculate_lcd(sz)
                sz1 = 7
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a7 = test.calculate_lcd(sz)
            elif self.orientation == 1 or self.orientation == 0:
                av, noise, homogeneity = test.water_roi_new()
                sz1 = 2
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a2 = test.calculate_lcd_new(sz)
                sz1 = 3
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a3 = test.calculate_lcd_new(sz)
                sz1 = 5
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a5 = test.calculate_lcd_new(sz)
                sz1 = 7
                sz = int(round(sz1 / ds.PixelSpacing[0]))
                a7 = test.calculate_lcd_new(sz)
            self.av = av  # 此处警告是上面条件语句没有完全考虑各种条件
            self.noise = noise * 100
            self.homogeneity = homogeneity
            self.lcd = a2
            self.water_lineEdit_CT.setText(self.strlen(av))
            self.water_lineEdit_noise.setText(self.strlen(noise * 100))
            self.water_lineEdit_junyun.setText(self.strlen(homogeneity))
            self.av2 = a2
            self.av3 = a3
            self.av5 = a5
            self.av7 = a7
            self.water_lineEdit_lcd_2.setText(self.strlen(a2));  # 不需要分号
            self.water_lineEdit_lcd_3.setText(self.strlen(a3));
            self.water_lineEdit_lcd_5.setText(self.strlen(a5));
            self.water_lineEdit_lcd_7.setText(self.strlen(a7));
            self.water_lineEdit_jz.setText('')
            print("the CT value of water phantom  is : ", av)
            print("the noise of water phantom is : %s %%" % (noise * 100))
            print("the homogeneity of water phantom is :", homogeneity)
            print("the lcd is:", a2)
            try:
                if self.ShowPos == True:
                    [xpos, ypos] = test.LabelPos
                    self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                    self.pix_label.fitInView1()  # ppaint.show()

            except:
                print("error in load LABELS in water phantom")
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '当前图像文件无法自动测量水模体相关指标！',
                                    QMessageBox.Ok)
            return

    def button_click_water_jz(self):
        # 均值和归一化
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        checkn = [0, 0, 0, 0]
        if self.water_checkbox1.isChecked():
            checkn[0] = 1
        if self.water_checkbox2.isChecked():
            checkn[1] = 1
        if self.water_checkbox3.isChecked():
            checkn[2] = 1
        if self.water_checkbox4.isChecked():
            checkn[3] = 1
        s1 = self.av2 * 2 / 10
        s2 = self.av3 * 3 / 10
        s3 = self.av5 * 5 / 10
        s4 = self.av7 * 7 / 10
        s5 = (s1 * checkn[0] + s2 * checkn[1] + s3 * checkn[2] + s4 * checkn[3]) / (sum(checkn))
        self.water_lineEdit_jz.setText(self.strlen(s5))

    def button_click_linear(self):
        QMessageBox.information(self,
                                '出错',
                                '儿童模体！',
                                QMessageBox.Ok)
        return
        fname = self.file_name;  # 语句无分号，return之后所有语句都不会执行，后面如果没用可以删掉
        fname = fname.replace('/', '\\\\')
        if (not self.C_Phantom) or (self.orientation != 2):
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体或非横截面图像！',
                                    QMessageBox.Ok)
            return
        try:
            test = Linearity_Phantom(fname)
            x = test.get_material_CT_values()
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '当前图像文件无法自动测量8种材料CT值！',
                                    QMessageBox.Ok)
            return
        # print(self.ShowPos)
        try:
            if self.ShowPos == True:
                [xpos, ypos] = test.LabelPos
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()

        except:
            print("error in load LABELS")
        ##        print(self.array1[0])
        self.array1 = sorted(x)
        self.linear_lineEdit_ctceliang1.setText("%.1f" % (self.array1[0]));  # 无分号
        self.linear_lineEdit_ctceliang2.setText("%.1f" % (self.array1[1]));
        self.linear_lineEdit_ctceliang3.setText("%.1f" % (self.array1[2]));
        self.linear_lineEdit_ctceliang4.setText("%.1f" % (self.array1[3]));
        self.linear_lineEdit_ctceliang5.setText("%.1f" % (self.array1[4]));
        self.linear_lineEdit_ctceliang6.setText("%.1f" % (self.array1[5]));
        self.linear_lineEdit_ctceliang7.setText("%.1f" % (self.array1[6]));
        self.linear_lineEdit_ctceliang8.setText("%.1f" % (self.array1[7]));
        self.linear_lineEdit_CT.setText('')
        self.CT_err = 0

    def button_click_slice8(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击层厚指标计算功能1',
        ##                                        QMessageBox.Ok)
        ##        return
        fname = self.file_name;  # 无分号，可以写成 fname = self.file_name.replace('','')格式
        fname = fname.replace('/', '\\\\')
        dcm = dicom.read_file(fname)
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        try:
            data_element = dcm.data_element("SliceThickness")
            self.slice8_biaocheng = data_element.value
            self.sliceThickness_lineEdit_bc8.setText(self.strlen(data_element.value))
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '未查询到层厚标称值，请检查图像！',
                                    QMessageBox.Ok)
            return
        try:
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom, diameter=75, pitch=self.pitch, number_beads=self.beadsnum)
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '当前图像文件无法自动测量层厚！',
                                    QMessageBox.Ok)
            return

        try:
            if self.orientation == 2:  # 横截面
                # thickness = spiralbeads.get_profile_transverse(displayImage=False)
                profile = spiralbeads.get_profile(displayImage=False)
                thickness = spiralbeads.get_lthickness(profile, bc=dcm.SliceThickness)
            elif self.orientation == 1:  # 冠状位
                profile = spiralbeads.locate_profile_coronal(displayImage=False)
                thickness = spiralbeads.get_thickness_coronal(profile)
            else:
                # self.orientation == 0:  # 矢状位
                profile = spiralbeads.locate_profile_sagittal(displayImage=False)
                thickness = spiralbeads.get_thickness_sagittal(profile)
            if thickness == None or thickness == 0:  # thickness有警告，上述条件有遗漏，可以把elif改成else
                QMessageBox.information(self,
                                        '出错',
                                        '当前图像无法测量层厚！',
                                        QMessageBox.Ok)
                return
        except ValueError:

            QMessageBox.information(self,
                                    '出错',
                                    '异常，无法测量层厚！',
                                    QMessageBox.Ok)
            return

        try:
            if self.ShowPos == True:
                [xpos, ypos] = spiralbeads.LabelPos
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()

        except:
            print("error in load LABELS")
        self.slice8_shice = thickness
        self.sliceThickness_lineEdit_shice8.setText(self.strlen(thickness))

        err1 = (thickness - data_element.value) / data_element.value * 100
        self.slice8_err = err1
        self.sliceThickness_lineEdit_error8.setText(self.strlen(err1))

    def button_click_slice28(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击层厚指标计算功能2',
        ##                                        QMessageBox.Ok)
        ##        return
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')  # 问题同上
        dcm = dicom.read_file(fname)
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        try:
            data_element = dcm.data_element("SliceThickness")
            self.slice28_biaocheng = data_element.value
            self.sliceThickness_lineEdit_bc28.setText(self.strlen(data_element.value))
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '未查询到层厚标称值，请检查图像！',
                                    QMessageBox.Ok)
            return
        try:
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom, diameter=75, pitch=self.pitch, number_beads=self.beadsnum)
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '当前图像文件无法自动测量层厚！',
                                    QMessageBox.Ok)
            return

        try:
            if self.orientation == 2:  # 横截面
                profile = spiralbeads.get_profile(displayImage=False)
                thickness = spiralbeads.get_lthickness(profile, bc=dcm.SliceThickness)
            elif self.orientation == 1:  # 冠状位
                profile = spiralbeads.locate_profile_coronal(displayImage=False)
                thickness = spiralbeads.get_thickness_coronal(profile)
            elif self.orientation == 0:  # 矢状位
                profile = spiralbeads.locate_profile_sagittal(displayImage=False)
                thickness = spiralbeads.get_thickness_sagittal(profile)
            if thickness == None or thickness == 0:  # 问题同上
                QMessageBox.information(self,
                                        '出错',
                                        '当前图像无法测量层厚！',
                                        QMessageBox.Ok)
                return
        except:

            QMessageBox.information(self,
                                    '出错',
                                    '异常，无法测量层厚！',
                                    QMessageBox.Ok)
            return

        try:
            if self.ShowPos == True:
                [xpos, ypos] = spiralbeads.LabelPos
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()

        except:
            print("error in load LABELS")
        self.slice28_shice = thickness
        self.sliceThickness_lineEdit_shice28.setText(self.strlen(thickness))
        ##        data_element = dcm.data_element("SliceThickness")
        ##        self.slice28_biaocheng = data_element.value
        ##        self.sliceThickness_lineEdit_bc28.setText(self.strlen(data_element.value))
        err1 = (thickness - data_element.value) / data_element.value * 100
        self.slice28_err = err1
        self.sliceThickness_lineEdit_error28.setText(self.strlen(err1))

    def button_click_slice2(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击层厚指标计算功能3',
        ##                                        QMessageBox.Ok)
        ##        return
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')  # 问题同上
        dcm = dicom.read_file(fname)
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        try:
            data_element = dcm.data_element("SliceThickness")
            self.slice2_biaocheng = data_element.value
            self.sliceThickness_lineEdit_bc2.setText(self.strlen(data_element.value))
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '未查询到层厚标称值，请检查图像！',
                                    QMessageBox.Ok)
            return
        try:
            phantom = CT_phantom(dcm)
            spiralbeads = SpiralBeads(phantom, diameter=75, pitch=self.pitch, number_beads=self.beadsnum)
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '当前图像文件无法自动测量层厚！',
                                    QMessageBox.Ok)
            return

        try:
            if self.orientation == 2:  # 横截面
                profile = spiralbeads.get_profile(displayImage=False)
                thickness = spiralbeads.get_lthickness(profile, bc=dcm.SliceThickness)
            elif self.orientation == 1:  # 冠状位
                profile = spiralbeads.locate_profile_coronal(displayImage=False)
                thickness = spiralbeads.get_thickness_coronal(profile)
            elif self.orientation == 0:  # 矢状位
                profile = spiralbeads.locate_profile_sagittal(displayImage=False)
                thickness = spiralbeads.get_thickness_sagittal(profile)
            if thickness == None or thickness == 0:  # 问题同上
                QMessageBox.information(self,
                                        '出错',
                                        '当前图像无法测量层厚！',
                                        QMessageBox.Ok)
                return
        except:

            QMessageBox.information(self,
                                    '出错',
                                    '异常，无法测量层厚！',
                                    QMessageBox.Ok)
            return

        try:
            if self.ShowPos == True:
                [xpos, ypos] = spiralbeads.LabelPos
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()

        except:
            print("error in load LABELS")
        self.slice2_shice = thickness
        self.sliceThickness_lineEdit_shice2.setText(self.strlen(thickness))
        ##        data_element = dcm.data_element("SliceThickness")
        ##        self.slice2_biaocheng = data_element.value
        ##        self.sliceThickness_lineEdit_bc2.setText(self.strlen(data_element.value))
        err1 = (thickness - data_element.value) / data_element.value * 100
        self.slice2_err = err1
        self.sliceThickness_lineEdit_error2.setText(self.strlen(err1))

    def button_click_ctxianxing(self):
        QMessageBox.information(self,
                                '出错',
                                '儿童模体！',
                                QMessageBox.Ok)
        return

        if (not self.C_Phantom) or (self.orientation != 2):#return之后不会执行一下语句，如果没用建议删掉
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体或非横截面图像！',
                                    QMessageBox.Ok)
            return
        try:
            self.a0 = float(self.linear_lineEdit_ctlilun1.text())
            self.a1 = float(self.linear_lineEdit_ctlilun2.text())
            self.a2 = float(self.linear_lineEdit_ctlilun3.text())
            self.a3 = float(self.linear_lineEdit_ctlilun4.text())
            self.a4 = float(self.linear_lineEdit_ctlilun5.text())
            self.a5 = float(self.linear_lineEdit_ctlilun6.text())
            self.a6 = float(self.linear_lineEdit_ctlilun7.text())
            self.a7 = float(self.linear_lineEdit_ctlilun8.text())
        except ValueError:
            QMessageBox.information(self,
                                    '出错',
                                    '输入的理论CT值有误，请重新输入！',
                                    QMessageBox.Ok)
            return
        ctarray = [0, 0, 0, 0, 0, 0, 0, 0]
        x = []
        y = []
        if self.checkbox1.isChecked():
            ctarray[0] = 1
            x.append(self.array1[0])
            y.append(self.a0)
        if self.checkbox2.isChecked():
            ctarray[1] = 1
            x.append(self.array1[1])
            y.append(self.a1)
        if self.checkbox3.isChecked():
            ctarray[2] = 1
            x.append(self.array1[2])
            y.append(self.a2)
        if self.checkbox4.isChecked():
            ctarray[3] = 1
            x.append(self.array1[3])
            y.append(self.a3)
        if self.checkbox5.isChecked():
            ctarray[4] = 1
            x.append(self.array1[4])
            y.append(self.a4)
        if self.checkbox6.isChecked():
            ctarray[5] = 1
            x.append(self.array1[5])
            y.append(self.a5)
        if self.checkbox7.isChecked():
            ctarray[6] = 1
            x.append(self.array1[6])
            y.append(self.a6)
        if self.checkbox8.isChecked():
            ctarray[7] = 1
            x.append(self.array1[7])
            y.append(self.a7)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        slope = round(slope, 3)
        intercept = round(intercept, 3)
        print(slope, intercept)

        def f(x, a, b):
            return a + b * x

        max2 = 0
        for i in range(len(x)):
            if abs(f(x[i], intercept, slope) - y[i]) > max2:
                max2 = abs(f(x[i], intercept, slope) - y[i])
            self.linear_lineEdit_CT.setText(self.strlen(max2))
            self.CT_err = max2

    def button_click_resolution_1(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击高对比分辨力——常规算法',
        ##                                        QMessageBox.Ok)
        ##        return
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        windwidth = 9
        if self.a8 != 0 and self.place == 1:
            try:
                bead = TungstenBead(
                    fname, roiwidth=32,
                    windowwidth=windwidth,
                    bead_position=[self.a8, self.a9]
                )
            except Exception as e:
                print(e)
                QMessageBox.information(self,
                                        '出错',
                                        '错误：' + str(e),
                                        QMessageBox.Ok)
                return
        else:
            try:
                bead = TungstenBead(
                    fname, roiwidth=32,
                    windowwidth=windwidth,
                    orientation_flag=self.orientation,
                )
                if bead.ISPOS == False:
                    QMessageBox.information(self,
                                            '出错',
                                            '错误：无法计算MTF！',
                                            QMessageBox.Ok)
                    return
            except Exception as e:
                print(e)
                QMessageBox.information(self,
                                        '出错',
                                        '错误：' + str(e),
                                        QMessageBox.Ok)
                return
        try:
            if self.ShowPos == True:
                [xpos, ypos] = [[bead.beadpositionx], [bead.beadpositiony]]
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()

        except:
            print("error in load LABELS")
        self.resolution10 = bead.resolution10 * 10
        self.resolution50 = bead.resolution50 * 10
        self.resolution_lineEdit_C_10.setText(self.strlen(bead.resolution10 * 10))
        self.resolution_lineEdit_C_50.setText(self.strlen(bead.resolution50 * 10))
        self.resolution_lineEdit_x.setText("%d" % (bead.beadpositionx))
        self.resolution_lineEdit_y.setText("%d" % (bead.beadpositiony))
        if self.resolution_checkbox1.isChecked():
            MTF_show(bead.resolution10 * 10, bead.resolution50 * 10, bead.freq * 10, bead.mtf)

    def button_click_resolution_2(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击高对比分辨力——高分辨算法',
        ##                                        QMessageBox.Ok)
        ##        return
        fname = self.file_name;
        fname = fname.replace('/', '\\\\')
        if not self.C_Phantom:
            QMessageBox.information(self,
                                    '出错',
                                    '非儿童模体！',
                                    QMessageBox.Ok)
            return
        windwidth = 9
        if self.a8 > 1 and self.place == 1:
            try:
                bead = TungstenBead(
                    fname, roiwidth=32,
                    windowwidth=windwidth,
                    bead_position=[self.a8, self.a9]
                )
            except Exception as e:
                print(e)
                QMessageBox.information(self,
                                        '出错',
                                        '错误：' + str(e),
                                        QMessageBox.Ok)
                return
        else:
            try:
                bead = TungstenBead(
                    fname, roiwidth=32,
                    windowwidth=windwidth,
                    orientation_flag=self.orientation,
                )
                if bead.ISPOS == False:
                    QMessageBox.information(self,
                                            '出错',
                                            '错误：无法计算MTF！',
                                            QMessageBox.Ok)
                    return
            except Exception as e:
                print(e)
                QMessageBox.information(self,
                                        '出错',
                                        '错误：' + str(e),
                                        QMessageBox.Ok)
                return
        try:
            if self.ShowPos == True:
                [xpos, ypos] = [[bead.beadpositionx], [bead.beadpositiony]]
                self.pix_label.loadLabels(xpos=xpos, ypos=ypos)
                self.pix_label.fitInView1()

        except:
            print("error in load LABELS")
        self.resolution10x = bead.resolution10 * 10
        self.resolution50x = bead.resolution50 * 10
        self.resolution_lineEdit_G_10.setText(self.strlen(bead.resolution10 * 10))
        self.resolution_lineEdit_G_50.setText(self.strlen(bead.resolution50 * 10))
        self.resolution_lineEdit_x.setText("%d" % (bead.beadpositionx))
        self.resolution_lineEdit_y.setText("%d" % (bead.beadpositiony))
        if self.resolution_checkbox2.isChecked():
            MTF_show(bead.resolution10 * 10, bead.resolution50 * 10, bead.freq * 10, bead.mtf)

    def button_click_saveReport(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击生成报表功能',
        ##                                        QMessageBox.Ok)
        ##        return
        mgy = self.mgy
        mgy1 = self.mgy1
        angle = self.angle
        av2 = self.av2
        av3 = self.av3
        av5 = self.av5
        av7 = self.av7
        noise = self.noise
        min1 = self.min1#函数内变量没用到，建议删掉
        max1 = self.max1
        mean1 = self.mean1
        lcd = self.lcd
        resolution = self.resolution
        homogeneity = self.homogeneity
        av = self.av
        slice_err8 = self.slice8_err
        slice_err28 = self.slice28_err
        slice2_err = self.slice2_err
        resolution10 = self.resolution10
        resolution50 = self.resolution50
        resolution10x = self.resolution10x
        resolution50x = self.resolution50x
        CT_err = self.CT_err
        orientation = ["（矢状位）", "（冠状位）", "（轴位）"]  # self.orientation

        styleBoldRed = xlwt.easyxf(
            'font: color-index black, bold off;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')  # 设置字体，颜色为黑色，加粗
        oldWb = xlrd.open_workbook("../template/template.xls", formatting_info=True)
        styleBold2 = xlwt.easyxf(
            'font:height 360,name Times New Roman,color-index black, bold on;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')
        newWb = copy1(oldWb)

        sheet1 = newWb.get_sheet(0)

        sheet1.write_merge(0, 1, 0, 6, "CT成像性能检测软件分析结果" + orientation[self.orientation], styleBold2)
        sheet1.write_merge(4, 4, 2, 2, self.strlen(self.slice8_biaocheng), styleBoldRed)
        sheet1.write_merge(4, 4, 3, 3, self.strlen(self.slice8_shice), styleBoldRed)
        sheet1.write_merge(4, 4, 4, 4, self.strlen(self.slice8_err), styleBoldRed)
        sheet1.write_merge(5, 5, 2, 2, self.strlen(self.slice28_biaocheng), styleBoldRed)
        sheet1.write_merge(5, 5, 3, 3, self.strlen(self.slice28_shice), styleBoldRed)
        sheet1.write_merge(5, 5, 4, 4, self.strlen(self.slice28_err), styleBoldRed)
        sheet1.write_merge(6, 6, 2, 2, self.strlen(self.slice2_biaocheng), styleBoldRed)
        sheet1.write_merge(6, 6, 3, 3, self.strlen(self.slice2_shice), styleBoldRed)
        sheet1.write_merge(6, 6, 4, 4, self.strlen(self.slice2_err), styleBoldRed)

        sheet1.write_merge(7, 7, 3, 3, self.strlen(angle), styleBoldRed)
        sheet1.write_merge(9, 11, 3, 3, self.strlen(av), styleBoldRed)
        sheet1.write_merge(12, 13, 3, 3, self.strlen(homogeneity), styleBoldRed)
        sheet1.write_merge(14, 15, 3, 3, self.strlen(noise), styleBoldRed)
        sheet1.write_merge(16, 18, 3, 3, self.strlen(resolution10), styleBoldRed)
        sheet1.write_merge(19, 21, 3, 3, self.strlen(resolution10x), styleBoldRed)
        sheet1.write_merge(22, 22, 3, 4, "HU=" + self.strlen(av2), styleBoldRed)
        sheet1.write_merge(23, 23, 3, 4, "HU=" + self.strlen(av3), styleBoldRed)
        sheet1.write_merge(24, 24, 3, 4, "HU=" + self.strlen(av5), styleBoldRed)
        sheet1.write_merge(25, 25, 3, 4, "HU=" + self.strlen(av7), styleBoldRed)
        sheet1.write_merge(26, 26, 3, 3, "%.1f" % (self.array1[0]), styleBoldRed)
        sheet1.write_merge(27, 27, 3, 3, "%.1f" % (self.array1[1]), styleBoldRed)
        sheet1.write_merge(28, 28, 3, 3, "%.1f" % (self.array1[2]), styleBoldRed)
        sheet1.write_merge(29, 29, 3, 3, "%.1f" % (self.array1[3]), styleBoldRed)
        sheet1.write_merge(30, 30, 3, 3, "%.1f" % (self.array1[4]), styleBoldRed)
        sheet1.write_merge(31, 31, 3, 3, "%.1f" % (self.array1[5]), styleBoldRed)
        sheet1.write_merge(32, 32, 3, 3, "%.1f" % (self.array1[6]), styleBoldRed)
        sheet1.write_merge(33, 33, 3, 3, "%.1f" % (self.array1[7]), styleBoldRed)

        file_name, ok = QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser(_fromUtf8('~/CT检测结果')),
            "xls (*.xls)")
        if file_name:
            newWb.save(file_name)
        newWb.save("0.xls")

    def button_click8(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击头部模体计算',
        ##                                        QMessageBox.Ok)
        ##        return

        s1 = (self.a72 + self.a73 + self.a74 + self.a75) / 6
        s2 = self.a71 / 3 + s1
        self.mgy = s2
        self.lineEdit_head_CTDIw.setText(self.strlen(s2, decimal_place=2))

    def button_click11(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击体部模体计算功能',
        ##                                        QMessageBox.Ok)
        ##        return
        s1 = (self.a82 + self.a83 + self.a84 + self.a85) / 6
        s2 = self.a81 / 3 + s1
        self.mgy1 = s2
        self.lineEdit_body_CTDIw.setText(self.strlen(s2, decimal_place=2))

    def open_directory(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setViewMode(QFileDialog.List)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            directory = dialog.selectedFiles()[0]
            ##            print(2)
            self.load_files(dicom_files_in_dir(directory))

    ##            print(1)
    def export_image(self):
        file_name, ok = QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser("~/dicom-export.png"),
            "PNG images (*.png)"
        )
        if file_name:
            self.pixmap_label._image.save(file_name)#没有属性

    def about(self):
        QMessageBox.information(self,
                                '关于',
                                '北京交通大学电子信息工程学院信号与图像处理实验室版权所有',
                                QMessageBox.Ok)

    def show_dock(self):
        # self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,self.file_dock)
        self.file_dock.show()

    def build_menu(self):
        self.file_menu = QMenu('&File', self)  # 文件
        self.file_menu.addAction('&Open Directory', self.open_directory, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('&File Dock', self.show_dock, QtCore.Qt.CTRL + QtCore.Qt.Key_L)
        self.file_menu.addAction('&Setting', self.change_data, QtCore.Qt.CTRL + QtCore.Qt.Key_D)
        self.file_menu.addAction('&Image Info', self.ImageInfo, QtCore.Qt.CTRL + QtCore.Qt.Key_S)  # 图片信息
        self.file_menu.addAction('&Save Report', self.button_click_saveReport, QtCore.Qt.CTRL + QtCore.Qt.Key_P)  # 生成报告
        self.file_menu.addAction('&Quit', self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)  # 退出
        self.help_menu = QMenu("&Help", self)
        self.help_menu.addAction('&About', self.about)  # 关于
        self.menuBar().addMenu(self.file_menu)
        self.menuBar().addMenu(self.help_menu)

    def ImageInfo(self):
        if self.file_name:
            dcm = dicom.read_file(self.file_name)
            print(str(dcm))
        try:
            data_element0 = dcm.data_element("SliceThickness")#条件不完善，如果不进入if条件，dcm无定义
            data_element1 = dcm.data_element("Manufacturer")
            data_element2 = dcm.data_element("KVP")
            data_element3 = dcm.data_element("XRayTubeCurrent")
            data_element4 = dcm.data_element("ExposureTime")
            data_element5 = dcm.data_element("ConvolutionKernel")
            QMessageBox.information(self,
                                    '图像信息',
                                    'Slice Thickness:     %f\nManufacturer:     %s\nKVP:     %s\nX-RayTubeCurrent:     %d\nExposureTime:     %s\nConvolutionKernel:     %s\nPixelSpacing:     %s\nReconstructionDiameter:     %s' % (
                                        data_element0.value, data_element1.value, data_element2.value,
                                        data_element3.value,
                                        data_element4.value, data_element5.value, dcm.PixelSpacing,
                                        dcm.ReconstructionDiameter),
                                    QMessageBox.Ok)
        except:
            QMessageBox.information(self,
                                    '错误',
                                    '部分信息未找到！',
                                    QMessageBox.Ok)

    def show_structure(self):
        if self.file_name:
            f = dicom.read_file(self.file_name)
            print(str(f))

    def toggle_full_screen(self, toggled):
        if toggled:
            self.setWindowState(QtCore.Qt.WindowFullScreen)
        else:
            self.setWindowState(QtCore.Qt.WindowNoState)

    def on_file_item_change(self):
        if not len(self.file_list.selectedItems()):
            self.file_name = None
        else:
            item = self.file_list.selectedItems()[0]
            self.file_name = item.toolTip()
            dcm = dicom.read_file(self.file_name)
            try:
                self.dcmKVP = dcm.data_element("KVP").value
                self.dcmXRayTubeCurrent = dcm.data_element("XRayTubeCurrent").value
                self.dcmExposureTime = dcm.data_element("ExposureTime").value
            except:
                QMessageBox.information(self,
                                        '提示',
                                        '部分信息未找到，请确认是否为模体图像！',
                                        QMessageBox.Ok)
                pass

    def load_files(self, files):
        self.file_list.clear()
        self.files = files
        for file_name in self.files:
            item = QListWidgetItem(os.path.basename(file_name))
            item.setToolTip(file_name)
            self.file_list.addItem(item)
        self.file_list.setMinimumWidth(self.file_list.sizeHintForColumn(0) + 20)
        if self.files:
            self.file_name = self.files[0]

    def get_coordinates(self, i, j):
        x = self.image_position[0] + self.pixel_spacing[0] * i #无定义
        y = self.image_position[1] + self.pixel_spacing[1] * j
        z = self.image_position[2]
        return x, y, z

    @property
    def mouse_ij(self):
        return self.mouse_y // self.zoom_factor, self.mouse_x // self.zoom_factor#无定义

    @property
    def mouse_xyz(self):
        if self.use_fractional_coordinates:
            correction = (self.zoom_factor - 1.) / (
                    2. * self.zoom_factor)
            return self.get_coordinates(self.mouse_x / self.zoom_factor - correction,
                                        self.mouse_y / self.zoom_factor - correction)
        else:
            return self.get_coordinates(self.mouse_x // self.zoom_factor, self.mouse_y // self.zoom_factor)

    def update_coordinates(self):
        if self.file:
            x, y, z = self.mouse_xyz
            i, j = self.mouse_ij
            self.z_label.setText("z: %.2f" % z)
            if i >= 0 and j >= 0 and i < self.data.shape[0] and j < self.data.shape[1]:
                self.resolution_lineEdit_x.setText("x: %.2f" % x)
                self.lineEdit_35.setText("y: %.2f" % y) #无定义
                self.ij_label.setText("Pos: (%d, %d)" % self.mouse_ij)
                self.hu_label.setText("HU: %d" % int(self.data[i, j]))
                return
            else:
                self.hu_label.setText("HU: ???")
        else:
            self.hu_label.setText("No image")

    def update_cw(self):
        self.update_coordinates()
        pass

    @property
    def file_name(self):
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        try:
            self._file_name = value
            # data = DicomData.from_files([self._file_name])
            self.pix_label.loadImage(self._file_name)  # loadImageFromFile(fileName = self._file_name)#.data = data
            self.C_Phantom = True
            dcm = dicom.read_file(self._file_name)
            n, self.orientation = image_orientation(dcm, orientation_flag=True)
            if self.orientation == 2:
                img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
                xc, yc, r = find_CT_phantom_outer_edge(img, dcm.PixelSpacing[0], return_coors=False)
                r1 = r * dcm.PixelSpacing[0]
                r2 = C_R
                for i in r2:
                    # print(abs(r1-i)*0.91)
                    if abs(r1 + 3 - i) < 4:  # abs(r1*0.91-i)<4
                        self.r = i
                print("R=" + str(r * dcm.PixelSpacing[0]))
                if r * dcm.PixelSpacing[0] < C_R[0] - 10 or r * dcm.PixelSpacing[0] > C_R[-1] + 10:
                    self.C_Phantom = False

            self.setWindowTitle(_translate("", "医用CT成像设备质量检测系统 - ", None) + self._file_name)
            self.place = 0
        except BaseException as exc:
            print(exc)
            self.pix_label.data = None
            self.setWindowTitle(_translate("", "医用CT成像设备质量检测系统 - No image", None))

    ##    def closeEvent(self.file_dock,event):
    ##        print "it's time to close window!!!"
    def center_selected(self, x=0, y=0):
        print(x, y)
        self.center_x = int(x)
        self.center_y = int(y)
        ##        print(self.center_x, self.center_y,self.pix_label.width(),self.pix_label.height())
        ##        if self.pix_label.height()<self.pix_label.width():
        ##            s = self.pix_label.height()
        ##            self.center_x = int(float(x/s)*510)
        ##            self.center_y = int(float(y/s)*510)
        ##        else:
        ##            s =self.pix_label.width()
        ##            self.center_x = int(float(x/s)*510)
        ##            self.center_y = int(float((y-(self.pix_label.height()-self.pix_label.width())/2)/s)*510)

        ##        info = self.center_x, ',', self.center_y

        reply = QMessageBox.information(self,
                                        '提示',
                                        '您确定用选取的坐标(%d,%d)作为图像中心吗？' % (self.center_x, self.center_y),
                                        QMessageBox.Yes | QMessageBox.No)
        if (reply == 16384):
            self.place = 1
            self.resolution_lineEdit_x.setText("%d" % (self.center_x))
            self.resolution_lineEdit_y.setText("%d" % (self.center_y))
            self.a8 = self.center_x
            self.a9 = self.center_y
        return

    def change_data(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '菜单设置项',
        ##                                        QMessageBox.Ok)
        vbox0 = QVBoxLayout()
        dialog0 = QDialog(self)
        dialog0.resize(200, 200)
        sForDataBtn = QPushButton("模体参数设置")
        sForCanvasBtn = QPushButton("画布显示设置")
        sCancelBtn = QPushButton("取消")
        vbox0.addWidget(sForDataBtn)
        ##        vbox0.addWidget(sForCanvasBtn)
        vbox0.addWidget(sCancelBtn)
        dialog0.setWindowTitle("SETTING")
        dialog0.setLayout(vbox0)

        def SettingData():
            """模体参数设置
            """
            fname = "thick.ini"  # *.ini
            dialog1 = QDialog(dialog0)
            DataEdit = QTextEdit()
            okBtn = QPushButton("应用")
            CcBtn = QPushButton("取消")
            vbox1 = QVBoxLayout()
            hbox1 = QHBoxLayout()
            hbox1.addWidget(okBtn)
            hbox1.addWidget(CcBtn)
            vbox1.addWidget(DataEdit)
            vbox1.addLayout(hbox1)
            dialog1.setLayout(vbox1)
            with open(fname, 'r', encoding='utf-8') as ffile:
                text = ffile.read()
                DataEdit.setText(text)

            def sChange():
                with open(fname, 'w', encoding='utf-8') as ffile1:
                    text = DataEdit.toPlainText()
                    ffile1.write(text)
                QMessageBox.information(self,
                                        '提示',
                                        '重启后生效！',
                                        QMessageBox.Ok)
                dialog1.close()

            def sCancel():
                dialog1.close()

            okBtn.clicked.connect(sChange)
            CcBtn.clicked.connect(sCancel)
            dialog1.exec_()
            return

        def SettingCanvas():
            """画布显示设置
            """
            fname = "Setting2.py"
            dialog2 = QDialog(dialog0)
            DGLabel = QLabel("图片背景")
            pix = QPixmap(QSize(30, 20))
            pix.fill(QtGui.QColor('red'))
            DGIcon = QIcon(pix)
            FLabel = QLabel("滚轮调速")
            FEdit = QLineEdit()
            WWLabel = QLabel("窗宽调速")
            WWEdit = QLineEdit()
            WCLabel = QLabel("窗位调速")
            WCEdit = QLineEdit()
            PenLabel = QLabel("标记画笔")
            ##            ComboBox2 = QComboBox()
            ##            ComboBox3 = QComboBox()
            ##            ComboBox4 = QComboBox()

            ##            DataEdit = QLineEdit()
            okBtn = QPushButton("应用")
            CcBtn = QPushButton("取消")
            gbox2 = QGridLayout()
            ##            hbox2 = QHBoxLayout()
            ##            hbox2.addWidget(okBtn)
            ##            hbox2.addWidget(CcBtn)

            gbox2.addWidget(DGLabel, 0, 0)
            gbox2.addWidget(DGIcon, 0, 1)
            gbox2.addWidget(FLabel, 1, 0)
            gbox2.addWidget(FEdit, 1, 1)
            gbox2.addWidget(WWLabel, 2, 0)
            gbox2.addWidget(WCLabel, 3, 0)
            gbox2.addWidget(PenLabel, 4, 0)
            gbox2.addWidget(okBtn, 5, 0)
            gbox2.addWidget(CcBtn, 5, 1)

            def sChange():
                with open(fname, 'w+', encoding='utf-8') as fout:
                    words = fout.readlines()
                    print("step1")
                    for line in words:
                        l = line.split("=", 1)
                        print(l)
                        if len(l) == 1 or line.startswith("if __name__ =="):
                            fout.write(line)
                            continue
                        print("step1")
                        if line.startswith("BACKGROUNDBRUSH"):
                            ##                            print(l)
                            m = l[0] + " = " + "QColor(30,30,30)" + "#" + l[1]
                            fout.write(m)
                            continue
                        print("step1")
                        if line.startswith("FACTOR0"):
                            ##                            print(l)
                            m = l[0] + " = " + "1" + "#" + l[1]
                            fout.write(m)
                            continue
                        print("step1")
                        if line.startswith("FACTOR1"):
                            ##                            print(l)
                            m = l[0] + " = " + "1" + "#" + l[1]
                            fout.write(m)
                            continue
                        print("step1")
                        if line.startswith("PEN"):
                            ##                            print(l)
                            m = l[0] + " = " + "QPen( Qt.red,3,Qt.SolidLine)" + "#" + l[1]
                            fout.write(m)
                            continue
                        print("step1")
                        if line.startswith("FACTOR"):
                            ##                            print(l)
                            m = l[0] + " = " + "1" + "#" + l[1]
                            fout.write(m)
                            continue
                        print("step5")
                        fout.write(line)
                        continue
                ##                    text = DataEdit.toPlainText()
                ##                    ffile1.write(text)
                QMessageBox.information(self,
                                        '提示',
                                        '重启后生效！',
                                        QMessageBox.Ok)
                dialog2.close()

            def sCancel():
                dialog2.close()

            okBtn.clicked.connect(sChange)
            CcBtn.clicked.connect(sCancel)
            # dialog2.setLayout(vbox2)
            dialog2.exec_()
            return

        def Cancel():
            """取消设置
            """
            reply = QMessageBox.information(self,
                                            '提示',
                                            '退出窗口吗？',
                                            QMessageBox.Ok)
            ##            print(reply)
            if reply == 1024:
                dialog0.close()

            return

        sForDataBtn.clicked.connect(SettingData)
        sForCanvasBtn.clicked.connect(SettingCanvas)
        sCancelBtn.clicked.connect(Cancel)
        dialog0.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog0.exec_()
        return

    def progress_wa(self):
        dia_pro = QDialog(self)
        dia_pro.resize(200, 200)
        vbox = QVBoxLayout()  # 纵向布局
        hbox = QHBoxLayout()  # 横向布局
        dia_pro.setWindowTitle("WAITING!")
        Progress = QProgressBar(dia_pro)
        #  self.yes =  QPushButton("OK")
        vbox.addWidget(Progress)
        # vbox.addWidget(self.yes)
        dia_pro.setLayout(vbox)
        # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)
        styleBoldRed = xlwt.easyxf(
            'font: color-index black, bold off;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')
        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet('wy_ws')
        ws.write_merge(0, 1, 0, 0, '分组', styleBoldRed)
        ws.write_merge(0, 1, 1, 1, '名称', styleBoldRed)
        ws.write_merge(0, 1, 2, 2, 'CT值（水）', styleBoldRed)
        ws.write_merge(0, 1, 3, 3, '噪声（%）', styleBoldRed)
        ws.write_merge(0, 1, 4, 4, '均匀性（HU）', styleBoldRed)
        ws.write_merge(0, 0, 5, 9, '低对比可探测能力', styleBoldRed)
        ws.write_merge(1, 1, 5, 5, '2', styleBoldRed)
        ws.write_merge(1, 1, 6, 6, '3', styleBoldRed)
        ws.write_merge(1, 1, 7, 7, '5', styleBoldRed)
        ws.write_merge(1, 1, 8, 8, '7', styleBoldRed)
        ws.write_merge(1, 1, 9, 9, 'LCD值', styleBoldRed)
        i = 2
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        completed = 0
        step = len(files)
        dia_pro.show()
        # self.dia_pro.exec_()
        for fname in files:
            completed += 1. / step * 100
            Progress.setValue(completed)
            dia_pro.show()
            f = fname  # str(fname)
            f = f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            try:
                ##                print(i)
                ds = dicom.read_file(fname)
                test = Water_Phantom(fname, r=self.r)
                n, self.orientation = image_orientation(ds, orientation_flag=True)
                if self.orientation == 2:
                    av, noise = test.water_roi()
                    homogeneity = test.homogeneity()
                    sz1 = 2
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a2 = test.calculate_lcd(sz)
                    sz1 = 3
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a3 = test.calculate_lcd(sz)
                    sz1 = 5
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a5 = test.calculate_lcd(sz)
                    sz1 = 7
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a7 = test.calculate_lcd(sz)
                elif self.orientation == 1 or self.orientation == 0:
                    av, noise, homogeneity = test.water_roi_new()
                    sz1 = 2
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a2 = test.calculate_lcd_new(sz)
                    sz1 = 3
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a3 = test.calculate_lcd_new(sz)
                    sz1 = 5
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a5 = test.calculate_lcd_new(sz)
                    sz1 = 7
                    sz = int(round(sz1 / ds.PixelSpacing[0]))
                    a7 = test.calculate_lcd_new(sz)

                s1 = a2 * 2 / 10#前面条件不完善，如果都不符合上述条件，无定义
                s2 = a3 * 3 / 10
                s3 = a5 * 5 / 10
                s4 = a7 * 7 / 10

                s5 = (s1 + s2 + s3 + s4) / 4

                ws.write(i, 0, q[0])
                ws.write(i, 1, q[-1])
                ws.write(i, 2, self.strlen(av))
                ws.write(i, 3, self.strlen(noise * 100))
                ws.write(i, 4, self.strlen(homogeneity))
                ws.write(i, 5, self.strlen(a2))
                ws.write(i, 6, self.strlen(a3))
                ws.write(i, 7, self.strlen(a5))
                ws.write(i, 8, self.strlen(a7))
                ws.write(i, 9, self.strlen(s5))
                i = i + 1
            except:
                i = i + 1
                continue
        file_name, ok = QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser(_fromUtf8('~/water')),
            "xls (*.xls)")
        if file_name:
            wb.save(file_name)
        dia_pro.close()

    def progress_th(self):
        ##        QMessageBox.information(self,
        ##                                        '提示',
        ##                                        '您正在点击统计列表中所有层厚指标功能',
        ##                                        QMessageBox.Ok)
        ##        return
        self.dia_pro = QDialog(self)
        self.dia_pro.resize(200, 200)
        vbox = QVBoxLayout()  # 纵向布局
        hbox = QHBoxLayout()  # 横向布局
        self.dia_pro.setWindowTitle("WAITING!")
        self.Progress = QProgressBar(self.dia_pro)
        #  self.yes =  QPushButton("OK")
        vbox.addWidget(self.Progress)
        # vbox.addWidget(self.yes)
        self.dia_pro.setLayout(vbox)
        # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)

        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet('wy_ws1')
        ws.write(0, 0, '分组')
        ws.write(0, 1, '名称')
        ws.write(0, 2, '标称')
        ws.write(0, 3, '实测')
        ws.write(0, 4, '误差')
        i = 1
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        self.completed = 0
        step = len(files)
        self.dia_pro.show()
        # self.dia_pro.exec_()
        for fname in files:
            self.completed += 1. / step * 100
            self.Progress.setValue(self.completed)
            self.dia_pro.show()
            f = fname  # str(fname)
            f = f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            # print fname
            dcm = dicom.read_file(fname)
            n, self.orientation = image_orientation(dcm, orientation_flag=True)
            if not self.orientation:
                continue

            try:

                phantom = CT_phantom(dcm)
                spiralbeads = SpiralBeads(phantom, diameter=75, pitch=self.pitch, number_beads=self.beadsnum)
                if self.orientation == 2:  # 横截面
                    profile = spiralbeads.get_profile(displayImage=False)
                    thickness = spiralbeads.get_lthickness(profile, bc=dcm.SliceThickness)
                elif self.orientation == 1:  # 冠状位
                    profile = spiralbeads.locate_profile_coronal(displayImage=False)
                    thickness = spiralbeads.get_thickness_coronal(profile)
                elif self.orientation == 0:  # 矢状位
                    profile = spiralbeads.locate_profile_sagittal(displayImage=False)
                    thickness = spiralbeads.get_thickness_sagittal(profile)
                else:
                    thickness = -1
            except:
                thickness = -1
            if thickness:
                err = (thickness - float(dcm.SliceThickness)) / float(dcm.SliceThickness) * 100
                ws.write(i, 0, q[0])
                ws.write(i, 1, q[-1])
                ws.write(i, 2, dcm.SliceThickness)
                ws.write(i, 3, self.strlen(thickness))
                ws.write(i, 4, self.strlen(err) + "%")
                i = i + 1
        file_name, ok = QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser(_fromUtf8('~/thickness')),
            "xls (*.xls)"
        )
        if file_name:
            wb.save(file_name)
        self.dia_pro.close()

    def progress_li(self):
        QMessageBox.information(self,
                                '出错',
                                '儿童模体！',
                                QMessageBox.Ok)
        return
        dia_pro = QDialog(self) #return之后语句建议删掉
        dia_pro.resize(200, 200)
        vbox = QVBoxLayout()  # 纵向布局
        hbox = QHBoxLayout()  # 横向布局
        dia_pro.setWindowTitle("WAITING!")
        Progress = QProgressBar(dia_pro)
        #  self.yes =  QPushButton("OK")
        vbox.addWidget(Progress)
        # vbox.addWidget(self.yes)
        dia_pro.setLayout(vbox)
        # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)
        styleBoldRed = xlwt.easyxf(
            'font: color-index black, bold off;borders:left THIN,right THIN,top THIN,bottom THIN;align:  vert center, horiz center')
        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet('wy_ws2')
        ws.write(0, 0, '分组')
        ws.write(0, 1, '名称')
        ws.write_merge(0, 0, 2, 9, '八种材料的测量CT值', styleBoldRed)

        i = 1
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        completed = 0
        step = len(files)
        dia_pro.show()
        # self.dia_pro.exec_()
        for fname in files:
            completed += 1. / step * 100
            Progress.setValue(completed)
            dia_pro.show()
            f = fname  # str(fname)
            f = f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            # print fname
            try:
                test = Linearity_Phantom(fname)
                x = test.get_material_CT_values()
                array1 = sorted(x)
                # print(array1[0])

                ws.write(i, 0, q[0])
                ws.write(i, 1, q[-1])
                ws.write(i, 2, "%.1f" % (array1[0]))
                ws.write(i, 3, "%.1f" % (array1[1]))
                ws.write(i, 4, "%.1f" % (array1[2]))
                ws.write(i, 5, "%.1f" % (array1[3]))
                ws.write(i, 6, "%.1f" % (array1[4]))
                ws.write(i, 7, "%.1f" % (array1[5]))
                ws.write(i, 8, "%.1f" % (array1[6]))
                ws.write(i, 9, "%.1f" % (array1[7]))
                i = i + 1
            except:
                i = i + 1
                continue
        file_name, ok = QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser(_fromUtf8('~/linear')),
            "xls (*.xls)"
        )
        if file_name:
            wb.save(file_name)
        dia_pro.close()

    def progress_MTF(self):
        dia_pro = QDialog(self)
        dia_pro.resize(200, 200)
        vbox = QVBoxLayout()  # 纵向布局
        hbox = QHBoxLayout()  # 横向布局
        dia_pro.setWindowTitle("WAITING!")
        Progress = QProgressBar(dia_pro)
        #  self.yes =  QPushButton("OK")
        vbox.addWidget(Progress)
        # vbox.addWidget(self.yes)
        dia_pro.setLayout(vbox)
        # self.dia_pro.setWindowModality(QtCore.Qt.ApplicationModal)

        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet('wy_ws2')
        ws.write(0, 0, '分组')
        ws.write(0, 1, '名称')
        ws.write(0, 2, '10%MTF(Lp/cm)')
        ws.write(0, 3, '50%MTF(Lp/cm)')

        i = 1
        self.ccdialog = 1
        self.chooseFile()
        if self.ccdialog == 1:
            return
        files = []
        for n in range(len(self.lname)):
            if self.lname[n] == 1:
                files.append(self.files[n])
        print(files)
        # files = self.files

        completed = 0
        step = len(files)

        dia_pro.show()
        # self.dia_pro.exec_()
        for fname in files:

            completed += 1. / step * 100
            Progress.setValue(completed)
            dia_pro.show()
            f = fname  # str(fname)
            f = f.split("/")
            q = f[-1].split('\\')
            fname = fname.replace('/', '\\\\')
            # print fname
            windwidth = 9
            n, self.orientation = image_orientation(fname, orientation_flag=True)
            try:
                bead = TungstenBead(
                    fname, roiwidth=32,
                    windowwidth=windwidth, orientation_flag=self.orientation,
                )
                ws.write(i, 0, q[0])
                ws.write(i, 1, q[-1])
                ws.write(i, 2, self.strlen(bead.resolution10 * 10))
                ws.write(i, 3, self.strlen(bead.resolution50 * 10))
                i = i + 1
            except:
                continue

        file_name, ok = QFileDialog.getSaveFileName(
            self,
            "Save file",
            os.path.expanduser(_fromUtf8('~/MTF')),
            "xls (*.xls)"
        )
        if file_name:
            wb.save(file_name)
        dia_pro.close()

    def chooseFile(self):
        dialog = QDialog(self)
        dialog.resize(200, 300)
        vBox = QVBoxLayout()  # 纵向布局
        groupBox = QGroupBox()
        groupBox1 = QGroupBox()
        vbox = QVBoxLayout(groupBox)  # 纵向布局
        vbox1 = QVBoxLayout(groupBox)  # 纵向布局
        hbox = [0] * len(self.files)
        cBox = [0] * len(self.files)
        label = [0] * len(self.files)
        self.lname = [0] * len(self.files)
        scroll = QScrollArea()

        for i in range(len(self.files)):
            hbox[i] = QHBoxLayout()  # 横向布局
            cBox[i] = QCheckBox()
            hbox[i].addWidget(cBox[i])
            f = str(self.files[i])
            f = f.split("/")
            q = f[-1].split('\\')
            label[i] = QLabel(q[-1])
            hbox[i].addWidget(label[i])
            vbox1.addLayout(hbox[i])

        groupBox1.setLayout(vbox1)
        scroll.setWidget(groupBox1)
        scroll.setAutoFillBackground(True)
        scroll.setWidgetResizable(True)

        self.select = 1
        selectBtn = QPushButton("全选")
        # vbox.addLayout(vbox1)
        vbox.addWidget(scroll)
        vbox.addWidget(selectBtn)
        dialog.setWindowTitle("选择列表文件")

        label_junyun1 = QLabel("均匀抽取n层")
        label_junyun2 = QLabel("   n = ")
        lineEdit_junyun = QLineEdit()
        okBtn1 = QPushButton("确定")
        hboxj = QHBoxLayout()  # 横向布局
        hboxj.addWidget(label_junyun1)
        hboxj.addWidget(label_junyun2)
        hboxj.addWidget(lineEdit_junyun)
        hboxj.addWidget(okBtn1)
        label_ge1 = QLabel("每n层抽取一层")
        label_ge2 = QLabel(" n = ")
        lineEdit_ge = QLineEdit()
        okBtn2 = QPushButton("确定")
        hboxg = QHBoxLayout()  # 横向布局
        hboxg.addWidget(label_ge1)
        hboxg.addWidget(label_ge2)
        hboxg.addWidget(lineEdit_ge)
        hboxg.addWidget(okBtn2)
        okBtn = QPushButton("确定")
        cancelBtn = QPushButton("取消")
        vbox.addLayout(hboxj)
        vbox.addLayout(hboxg)
        hboxs = QHBoxLayout()  # 横向布局
        hboxs.addWidget(okBtn)
        hboxs.addWidget(cancelBtn)
        vBox.addWidget(groupBox)
        vBox.addLayout(hboxs)
        dialog.setLayout(vBox)

        def cbchange():
            for i in range(len(self.files)):
                if (cBox[i].isChecked()):
                    self.lname[i] = 1
                else:
                    self.lname[i] = 0
            if self.lname == [0] * len(self.files):
                selectBtn.setText("全选")
                self.select = 1
            elif self.lname == [1] * len(self.files):
                selectBtn.setText("全部取消")
                self.select = 0
            else:
                pass

        def canc():
            dialog.close()

        # self.ccdialog = 1
        def cBoxall():
            if self.select == 1:
                for i in range(len(self.files)):
                    cBox[i].setCheckState(QtCore.Qt.Checked)  # cBox[i].toggle()
                selectBtn.setText("全部取消")
                self.select = 0
            else:
                for i in range(len(self.files)):
                    cBox[i].setCheckState(QtCore.Qt.Unchecked)  # cBox[i].toggle()
                selectBtn.setText("全选")
                self.select = 1

        def togbox():
            if self.lname == [0] * len(self.files):
                QMessageBox.information(self,
                                        '出错',
                                        '请选择文件！',
                                        QMessageBox.Ok)
                self.ccdialog = 1
            else:
                self.ccdialog = 0
                dialog.close()

        def cBoxj():
            for i in range(len(self.files)):
                cBox[i].setCheckState(QtCore.Qt.Unchecked)
            try:
                nj = int(lineEdit_junyun.text())

                n = 0  # 设置n用于防止实际抽取数量大于设定值
                for i in range(0, len(self.files), int(len(self.files) / nj)):
                    n = n + 1
                    # print i
                    if n > nj: break
                    cBox[i].toggle()
            except:
                QMessageBox.information(self,
                                        '出错',
                                        '请输入正确数值！',
                                        QMessageBox.Ok)

        def cBoxg():
            for i in range(len(self.files)):
                cBox[i].setCheckState(QtCore.Qt.Unchecked)
            try:
                ng = int(lineEdit_ge.text())
                print(ng)
                for i in range(0, len(self.files), ng):
                    cBox[i].toggle()
            except:
                QMessageBox.information(self,
                                        '出错',
                                        '请输入正确数值！',
                                        QMessageBox.Ok)

        for i in range(len(self.files)):
            cBox[i].stateChanged.connect(
                cbchange)  # QtCore.QObject.connect(cBox[i], QtCore.SIGNAL(_fromUtf8('stateChanged(int)')), cbchange)
        cancelBtn.clicked.connect(
            canc)  # QtCore.QObject.connect(cancelBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), canc)
        selectBtn.clicked.connect(
            cBoxall)  # QtCore.QObject.connect(selectBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), cBoxall)
        okBtn1.clicked.connect(cBoxj)  # QtCore.QObject.connect(okBtn1, QtCore.SIGNAL(_fromUtf8("clicked()")), cBoxj)
        okBtn2.clicked.connect(cBoxg)  # QtCore.QObject.connect(okBtn2, QtCore.SIGNAL(_fromUtf8("clicked()")), cBoxg)
        okBtn.clicked.connect(togbox)  # QtCore.QObject.connect(okBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), togbox)

        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.exec_()

    def cBoxchange(self):
        if (self.cBox.isChecked()):#属性无定义
            self.cb = 1
            self.lineEdit_dia.setEnabled(False)
            self.lineEdit_pit.setEnabled(False)
            self.lineEdit_n.setEnabled(False)
        else:
            self.cb = 0
            self.lineEdit_dia.setEnabled(True)
            self.lineEdit_pit.setEnabled(True)
            self.lineEdit_n.setEnabled(True)

    def readData(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File')
        n = str(fname)
        if (not (os.path.exists(fname) and os.path.isfile(fname))):
            print("file does not exist!")
        else:
            f = open(fname, 'r')
            for i in f:
                m = i.split()
            #  print m
            f.close()
            self.lineEdit_dia.setText(m[0])
            self.lineEdit_pit.setText(m[1])
            self.lineEdit_n.setText(m[2])

    def saveData(self):
        file_name, ok = QFileDialog.getSaveFileName(self,
                                                    "Save file",
                                                    os.path.expanduser("~/beadsData"))
        if file_name:
            with open(file_name, 'w') as f:
                s = str(self.lineEdit_dia.text()) + " " + str(self.lineEdit_pit.text()) + " " + str(
                    self.lineEdit_n.text())
                #  print s
                f.write(s)
                f.close()

    def ok(self):
        print("应用！")
        self.diam = float(self.lineEdit_dia.text())
        self.pitch = float(self.lineEdit_pit.text())
        self.beadsnum = int(self.lineEdit_n.text())
        self.dialog.close()

    def cancel(self):
        print("取消！")
        self.dialog.close()

    def strlen(self, s, decimal_place=0):
        s = ("{0:.3f}".format(s))
        s1 = float(s)
        # print s1,len(s)
        if abs(s1) < 1 and decimal_place == 0:  # && len(s) ==  5:
            return s
        elif abs(s1) < 10 or decimal_place == 2:
            s = "%.2f" % s1
        elif abs(s1) < 100 or decimal_place == 1:
            s = "%.1f" % s1
        else:
            s = str(int(s1))
        return (s)  # "%.3f"% s))


if __name__ == "__main__":
    import sys

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt4.QtGui import QApplication
        except ImportError:
            raise ImportError("QtImageViewer: Requires PyQt5 or PyQt4.")
    print(('Using Qt ' + QT_VERSION_STR))
    width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    ##    print(width,height)
    hDC = win32gui.GetDC(0)
    dpi = win32print.GetDeviceCaps(hDC, win32con.LOGPIXELSX)
    print(width, height, dpi)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    if len(sys.argv) < 2:
        path = ".."
    else:
        path = sys.argv[1]
    ui = Viewer(path)
    ui.show()
    sys.exit(app.exec_())
