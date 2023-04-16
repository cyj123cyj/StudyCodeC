# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import pydicom
import numpy as np
from scipy import misc
import copy
from settings.Setting import BACKGROUNDBRUSH, FACTOR, PEN  # *
from utils.util import image_orientation


class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        """画布初始化，场景self._scene，以添加Item的方式控制显示，self._photo为图像，self.ppaint为标记
        """
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self.ppaint = MyPaintItem()

        self._scene.addItem(self._photo)
        self._scene.addItem(self.ppaint)

        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(BACKGROUNDBRUSH))  # 灰色 背景设置QtGui.QColor(30, 30, 30)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.data = None
        self.parent = parent
        self.factor = 1
        self.setCursor(QtCore.Qt.CrossCursor)

    def hasPhoto(self):
        """是否显示图像，与self._empty值相反
        """
        return not self._empty

    def fitInView1(self, scale=True):
        """使图像显示适应窗口
        """
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                ##                print (factor)
                self.scale(factor, factor)
            self._zoom = 0

    ##            self.factor = 1

    def setPhoto(self, pixmap=None):
        """初始图像显示，并适应窗口范围
        """
        self._zoom = 0
        if pixmap and not pixmap.isNull():

            self._empty = False
            # self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.setCursor(QtCore.Qt.CrossCursor)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.setCursor(QtCore.Qt.CrossCursor)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView1()
        self.loadLabels(xpos=[], ypos=[])

    def wheelEvent(self, event):
        """鼠标滚轮事件，向下滚动放大画布，向上滚动缩小画布，适应窗口时self._zoom = 0
        """
        # self.factor = 1
        if self.hasPhoto():
            event.accept()
            if event.angleDelta().y() > 0:
                factor = 1.25 * FACTOR
                self._zoom += 1
            else:
                factor = 1 / 1.25 / (FACTOR)  # 用户设置，调速，用户偏好常数
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
                self.factor = factor
            elif self._zoom == 0:
                self.fitInView1()
                self.factor = 1
            else:
                self._zoom = 0

    ##            print(self.viewport().geometry(),self.sceneRect(),self.viewport().rect(),self._scene.itemsBoundingRect())
    ##            print("here")
    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        if self.hasPhoto():
            ##            self.scale(1,1)
            ##            return
            rect = QtCore.QRectF(self._photo.pixmap().rect())
            ##        print(rect)
            if not rect.isNull():
                self.setSceneRect(rect)
                if self.hasPhoto():
                    unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                    ##                    self.scale(1 / unity.width(), 1 / unity.height())
                    viewrect = self.viewport().rect()
                    scenerect = self.transform().mapRect(rect)
                    factor = min(viewrect.width() / scenerect.width(),
                                 viewrect.height() / scenerect.height())
                    ##                    print(self._zoom)
                    if self._zoom == 0:
                        self.scale(factor, factor)
                    else:
                        self.scale(factor * self._zoom, factor * self._zoom)


    def toggleDragMode(self):
        """切换画布拖动模式
        """
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        """鼠标点击事件，左键+ctrl键获取坐标，左键单击拖动画布，右键单击记录当前位置
        """
        super(PhotoViewer, self).mousePressEvent(event)
        if not self.hasPhoto():
            return
        ##        print("01111")
        ##        point = self.mapToScene(event.pos())
        ##        print(point.x(),point.y())
        ##        point = self.mapToScene(self.mapFromParent(event.pos()))
        ##        print(point.x(),point.y())
        if event.button() == QtCore.Qt.LeftButton and event.modifiers() & QtCore.Qt.ControlModifier:  # event.modifiers() == QtCore.Qt.Key_Control:
            rect = QtCore.QRectF(self._photo.pixmap().rect())
            # viewrect = self.viewport().rect()
            # scenerect = self.transform().mapRect(rect)
            point = self.mapToScene(event.pos())  # self.mapToScene(self.mapFromParent(event.pos()))

            x0 = point.x()
            y0 = point.y()
            print(x0, y0)

            ##            self.parent.center_selected(x=x0,y=y0)

            reply = QtWidgets.QMessageBox.information(self,
                                                      '提示',
                                                      '您确定用选取的坐标(%d,%d)作为图像中心吗？' % (x0, y0),
                                                      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            print(reply)
            try:
                if (reply == 16384):
                    self.parent.place = 1
                    self.parent.resolution_lineEdit_x.setText("%d" % (x0))
                    self.parent.resolution_lineEdit_y.setText("%d" % (y0))
                    self.parent.a8 = int(x0)
                    self.parent.a9 = int(y0)
            except:
                pass
        elif event.button() == QtCore.Qt.LeftButton:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            ##            self.setCursor(QtCore.Qt.CrossCursor)
            if self._photo.isUnderMouse():
                self.photoClicked.emit(QtCore.QPoint(event.pos()))
            super(PhotoViewer, self).mousePressEvent(event)
        ##            self.LeftMouse = True
        ##            self.RightMouse = False
        elif event.button() == QtCore.Qt.RightButton:
            ##            print(event.x(),event.y())

            self.mouse_x = event.x()  # 当前鼠标位置
            self.mouse_y = event.y()


    def mouseMoveEvent(self, event):
        ##        print (event)
        """鼠标移动事件，右键禁止拖动画布，并记录移动距离来调整窗宽窗位
        """
        if not self.hasPhoto():
            return
        super(PhotoViewer, self).mouseMoveEvent(event)
        try:
            point = self.mapToScene(event.pos())
            i = int(point.x())
            j = int(point.y())
            ##            print ("Pos:[%d,%d],HU: %d" %(int(i) ,int(j) ,int(self.data[i, j])))
            self.parent.hu_label.setText("Pos:[%d,%d],HU: %d" % (int(i), int(j), int(self.data[j, i])))
        except:
            pass
        if event.buttons() == QtCore.Qt.RightButton:  # self.RightMouse :#event.button() == QtCore.Qt.RightButton:
            ##            print("right")
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.x_move = event.x() - self.mouse_x
            self.y_move = event.y() - self.mouse_y
            if abs(self.x_move) < 2 and abs(self.y_move) < 2:
                return
            self.genImage(self.x_move, self.y_move)  # 调节窗宽窗位

    def genImage(self, x, y):
        """根据传入的x、y变化量调整窗宽窗位
        """
        ##        print(self._scene.width,self._scene.height)

        rect = QtCore.QRectF(self._photo.pixmap().rect())
        viewrect = self.viewport().rect()
        scenerect = self.transform().mapRect(rect)
        try:
            FACTOR0 = self.parent.speedSlider.value() / 5
            FACTOR1 = self.parent.speedSlider.value() / 5
        except:
            pass
        # print(viewrect,scenerect,x,y)
        x = x / scenerect.width() * 256  # self.window_center
        y = y / scenerect.height() * 256  # self.window_width
        ww = self.h - self.l + y / 20 * FACTOR0  # 窗宽 ,如果进入except中无Factory定义
        wc = (self.h + self.l) / 2 + x / 20 * FACTOR1  # 窗位
        if ww < 1:
            ww = 1  # 这样不影响窗位调节。
        self.l = wc - ww / 2
        self.h = wc + ww / 2  # self.h+y/10

        try:
            self.parent.wwcCheckbox.setCheckState(QtCore.Qt.Unchecked)
            self.parent.KeepWindowSetting = False
            self.parent.lineEdit_ww.setText(str(int(self.h - self.l)))
            self.parent.lineEdit_wc.setText(str(int((self.h + self.l) / 2)))
        except:
            pass

        data = (self.data - self.l) / (self.h - self.l) * 256
        data[data < 0] = 0
        data[data > 255] = 255
        data = data.astype("int8")

        image = QtGui.QImage(data, data.shape[1], data.shape[0], data.shape[1], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap.fromImage(image)
        ##        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        ##        self._empty = False
        ##        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self._photo.setPixmap(pixmap)  # self.setPhoto(pixmap)

    def mouseDoubleClickEvent(self, event):
        """鼠标双击事件:右键读入原始像素数据，调整窗宽窗位至默认值；左键恢复图像大小，使适应屏幕
        """
        if not self.hasPhoto():
            return
        super(PhotoViewer, self).mouseDoubleClickEvent(event)

        if event.button() == QtCore.Qt.RightButton:
            ##            print("double")
            data = copy.deepcopy(self.data)

            self.l = self.window_center - self.window_width / 2  # 0
            self.h = self.window_center + self.window_width / 2  # 255
            ##            try:
            ##                if self.parent.KeepWindowSetting == True:
            ##                    self.l = int(self.parent.lineEdit_wc.text())- int(self.parent.lineEdit_ww.text())/2#self.parent.l
            ##                    self.h = int(self.parent.lineEdit_wc.text())+ int(self.parent.lineEdit_ww.text())/2#self.parent.h
            ##            except:
            ##                pass
            data = (data - self.l) / (self.h - self.l) * 256
            data[data < 0] = 0
            data[data > 255] = 255
            data = data.astype("int8")
            try:
                self.parent.lineEdit_ww.setText(str(int(self.h - self.l)))
                self.parent.lineEdit_wc.setText(str(int((self.h + self.l) / 2)))
            except:
                pass
            image = QtGui.QImage(data, data.shape[1], data.shape[0], data.shape[1], QtGui.QImage.Format_Indexed8)
            pixmap = QtGui.QPixmap.fromImage(image)
            self._photo.setPixmap(pixmap)  # self.setPhoto(pixmap)
        elif event.button() == QtCore.Qt.LeftButton:
            self.fitInView1()

    def loadImage(self, fileName=""):
        """读入图像，初始显示依据tag中的窗宽窗位值
        """
        if len(fileName) and os.path.isfile(fileName):
            dcm = pydicom.read_file(fileName, force=True)

            img = dcm.pixel_array
            try:
                self.window_center = dcm.WindowCenter[0]
                self.window_width = dcm.WindowWidth[0]
            except:
                self.window_center = dcm.WindowCenter
                self.window_width = dcm.WindowWidth

            self.l = self.window_center - self.window_width / 2  # 0
            self.h = self.window_center + self.window_width / 2  # 255
            print(self.l, self.h)
            try:
                if self.parent.KeepWindowSetting == True:
                    self.l = int(self.parent.lineEdit_wc.text()) - int(
                        self.parent.lineEdit_ww.text()) / 2  # self.parent.l
                    self.h = int(self.parent.lineEdit_wc.text()) + int(
                        self.parent.lineEdit_ww.text()) / 2  # self.parent.h
                self.parent.lineEdit_ww.setText(str(int(self.h - self.l)))
                self.parent.lineEdit_wc.setText(str(int((self.h + self.l) / 2)))
                self.parent.hu_label.setText(fileName)
                n, ind = image_orientation(dcm, orientation_flag=True)
                ##                print(ind)
                if ind == 2:
                    orientation = "横截面"
                elif ind == 1:
                    orientation = "冠状位"
                elif ind == 0:
                    orientation = "矢状位"
                self.parent.z_label.setText(orientation)#条件不完善
            except:
                pass

            ##
            try:
                data = np.dot(dcm.RescaleSlope, img) + dcm.RescaleIntercept
                self.data = copy.deepcopy(data)  ###
                a = self.l  # -1000
                data = (data - self.l) / (self.h - self.l) * 256
                # self.data = data
                data[data < 0] = 0
                data[data > 255] = 255
                data = data.astype("int8")
            except:
                data = img
                data = np.squeeze(data)
                data = misc.bytescale(data)

            image = QtGui.QImage(data, data.shape[1], data.shape[0], data.shape[1], QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.setPhoto(pixmap)
        print(("Reading %s..." % fileName))

    def loadLabels(self, xpos=(200, 400, 600), ypos=(200, 400, 600), posn=((1, 1), (20, 20), (50, 50), (512, 512))):
        self.ppaint.xpos = list(xpos)
        self.ppaint.ypos = list(ypos)
        self.ppaint.pos = list(posn)


# self.ppaint.paint()

class MyPaintItem(QtWidgets.QGraphicsItem):
    def __init__(self):
        super(MyPaintItem, self).__init__()
        self.xpos = []  # 1,10,30,50,100]
        self.ypos = []  # 1,10,30,50,100]
        self.pos = []

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 512, 512)

    def paint(self, painter, option, widget): #建议改名称

        #  points = [QtCore.QPoint(10, 10), QtCore.QPoint(20, 20), QtCore.QPoint(30, 30), QtCore.QPoint(200, 200)]
        points = []
        painter.setPen(PEN)
        painter.drawPoints(QtGui.QPolygon(points))
        if len(self.xpos) == 0:  # self.xpos
            return
        for i in range(len(self.xpos)):  # self.xpos
            points.append(QtCore.QPoint(int(self.xpos[i]),
                                        int(self.ypos[i])))  # self.xpos[i],self.ypos[i]#self.pos[i][0],self.pos[i][1]
        ##        print(PEN)
        # QtGui.QColor(166,66,250))##.NoPen)##QtGui.QPen(QtCore.Qt.red,5,QtCore.Qt.SolidLine)
        painter.setBrush(QtCore.Qt.red)  ##.darkGray)
        painter.drawPoints(QtGui.QPolygon(points))


class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = PhotoViewer(parent=self)
        self.viewer.resize(300, 300)
        # 'Load image' button
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        # Button to change from drag/pan to getting pixel info
        self.btnPixInfo = QtWidgets.QToolButton(self)
        self.btnPixInfo.setText('Enter pixel info mode')
        self.btnPixInfo.clicked.connect(self.pixInfo)
        self.editPixInfo = QtWidgets.QLineEdit(self)
        self.editPixInfo.setReadOnly(True)
        self.viewer.photoClicked.connect(self.photoClicked)
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnPixInfo)
        HBlayout.addWidget(self.editPixInfo)
        VBlayout.addLayout(HBlayout)

    def loadImage(self):
        ##        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]
        ##        print(name)
        fileName, filetype = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                   "选取文件",
                                                                   "./",
                                                                   "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
        print(fileName)
        ##        name = name.replace('/', '\\\\')
        self.viewer.loadImage(fileName)
        self.viewer.loadLabels()

    # self.viewer.setPhoto(QtGui.QPixmap(r'biaochengguanxi.jpg'))

    def pixInfo(self):
        # from Setting2 import *
        print("ok")
        self.viewer.loadLabels()
        ##        self.viewer.setScene(self.viewer._scene)
        ##        super(PhotoViewer, self.viewer).__init__(self)
        print(BACKGROUNDBRUSH)
        self.viewer.toggleDragMode()

    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    ##    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())
