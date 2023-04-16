import scipy
import numpy
from scipy.ndimage import label  # 多维图像处理
from scipy.optimize import curve_fit  # 指数幂数函数拟合确定参数

##from scipy.stats import norm#multivariate_normal
from math import floor, sqrt, tan  # 浮点数向上取整，求平方根
import pylab
import re
from calculation_logic.linear import find_rod_locations
from calculation_logic.resolution import roi_generator
from utils.util import find_CT_phantom_outer_edge, canny, find_edge_new

# 读取配置文件
import configparser
import os

curpath = os.path.dirname(os.path.realpath(__file__))
cfgpath = os.path.join(curpath, "../thick.ini")
print('ini文件的路径:', cfgpath)  # ini文件的路径
conf = configparser.ConfigParser()  # 创建管理对象
conf.read(cfgpath, encoding="utf-8-sig")  # 读ini文件
sections = conf.sections()  # 获取所有的section
# print('ini文件的section：',sections)  #返回list

theta_method = conf.items('thickness_theta')  # 某section中的内容
# print('thickness_theta部分：',theta_method)  #list里面对象是元组

number_samples = int(theta_method[0][1])  # 25000,圆环上取样点数
diameter = float(theta_method[1][1])  # 75，钨珠所在圆环的直径（mm）
pitch = float(theta_method[2][1])  # 90，螺距
number_beads = int(theta_method[3][1])  # 180，一个螺距上的珠子数


def alphanum_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def Gauss(x, a, x0, sigma, b):  # 数据拟合所用高斯函数公式，a指高斯曲线的峰值，x0为其对应的横坐标，sigma为标准差，b为背景值
    return a * numpy.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def fit_Gauss(x, y):
    mn = x[numpy.argmax(y)]  # max(y)所对应的横坐标x
    sig = sqrt(sum(y * (x - mn) ** 2) / sum(y))
    if sig > 0.5:
        sig = 0.5  # 标准差
    bg = min(y)  # y的最小值
    try:
        popt, pcov = curve_fit(Gauss, x, y, p0=[max(y), mn, sig, bg])  # 高斯拟合，拟合序列：x，y；Gauss参数：p0
        print("mn=")
        print(mn, numpy.argmax(y), y, sig)  # 高斯函数公式的参数
    except RuntimeError:
        return None
    return popt, pcov  # Gauss公式所需参数


#####横截面的层厚计算 Transverse plane
class SpiralBeads:
    # phantom geometry
    # key: value = spiral bead pitch: number of beads for a full 2pi circle
    # the sprial bead pitch is in millimeter.

    # modified the parameter list so that only one group of beads
    #   is used for slice thickness calculation
    ##    def __init__(self, phantom, type_A=False, **kwargs):
    def __init__(self, phantom,  # 待计算的综合模体
                 diameter=166.3,  # 设置：钨珠所在圆环的直径（mm）
                 pitch=90,  # 设置：螺距
                 number_beads=180,  # 设置：一个螺距上的珠子数
                 interval=75,  # 设置：两钨丝间平行距离（mm）
                 dip=26.56,  # 设置：钨丝与水平方向的夹角
                 **kwargs):
        self.phantom = phantom
        self.sliceThickness = self.phantom.dicom.SliceThickness  # 层厚标称值
        print('层厚标称值 = ', self.sliceThickness)
        self.number_samples = number_samples  # 5000*5  #圆环上的采样点个数

        self.rou = floor(diameter / 2.0 / self.phantom.dicom.PixelSpacing[0])  # 半径/像素间距，由物理尺寸计算像素阵尺寸
        self.pitch = pitch

        self.phantom_high = 94  # 冠状面综合模94mm
        self.whole_high = 198  # 水模加综合模
        self.dis_sagittal = 25
        self.dip = dip * scipy.pi / 180
        self.interval = interval / 2.0 / self.phantom.dicom.PixelSpacing[0]  # 半径/像素间距，由物理尺寸计算像素阵尺寸
        self.High = self.phantom_high / self.phantom.dicom.PixelSpacing[0]
        self.whole = self.whole_high / self.phantom.dicom.PixelSpacing[0]
        self.dis_sagi = self.dis_sagittal / self.phantom.dicom.PixelSpacing[0]  # 像素尺寸
        self.move_left = 0  # 记录像素值曲线的平移
        self.move_right = 0
        self.ROI_A = []  # 像素值曲线向两侧展宽的范围

        x0, y0, xs, ys, x1, x2, y1, y2 = find_edge_new(self.phantom.image, self.phantom.dicom.PixelSpacing[0],
                                                       returnWaterEdge=False)
        self.x0 = x0  # 冠状面和矢状面模体中心横坐标
        self.y0 = y0
        self.dis2radRatio = None
        self.profile = self.get_profile()
        self.LabelPos = None  # 存储钨丝位置坐标

    def smooth_curve(self, curve, w=100):  # 平滑像素值曲线
        curve_m = scipy.ndimage.filters.maximum_filter(curve, w)
        curve_mg = scipy.ndimage.filters.gaussian_filter1d(curve_m, w)
        curve_mgm = scipy.ndimage.filters.maximum_filter(curve_mg, w)
        return curve_mg, curve_mgm

    def get_profile_coro_sagi(self, displayImage=False):  ######冠状面、矢状面获得左右两边的钨丝所在的原始像素值曲线
        pa = self.phantom.image  # 模体图像CT值矩阵
        dis = self.interval  # 以像素个数为单位的半径
        space = self.phantom.dicom.PixelSpacing[0]
        y = []
        for i in range(0, self.number_samples):
            y.append(self.High * i / self.number_samples)  # 钨丝所在位置，取样点纵坐标
        y = y + self.y0 - self.whole
        profile_left_off = []  # 左侧钨丝
        profile_right_off = []
        offsets = [-1, 0, 1]  # 圆环半径偏差值
        # print(len(y),len(x_left),pa.shape)
        for off in offsets:
            x_left = numpy.ones(self.number_samples) * (self.x0 - dis + off)  # 取样点横坐标
            x_right = numpy.ones(self.number_samples) * (self.x0 + dis + off)
            pf_left = scipy.ndimage.map_coordinates(pa, numpy.vstack((y, x_left)),  # 将pa中坐标为(y,x)的像素提取出来
                                                    order=3, mode='wrap')
            pf_right = scipy.ndimage.map_coordinates(pa, numpy.vstack((y, x_right)),  # 将pa中坐标为(y,x)的像素提取出来
                                                     order=3, mode='wrap')
            profile_left_off.append(pf_left)
            profile_right_off.append(pf_right[::-1])  # 右侧像素值曲线反转
        profile_left_off = numpy.array(profile_left_off)
        profile_right_off = numpy.array(profile_right_off)

        profile_left = profile_left_off.max(axis=0)  # 综合三条像素值曲线得到一条，同一位置选择最大像素值
        profile_right = profile_right_off.max(axis=0)
        self.ROI_A = [profile_left_off, profile_right_off]

        if displayImage:
            pylab.figure()
            pylab.imshow(pa, interpolation="nearest", cmap='gray')  # 原始模体图像
            pylab.plot(x_left, y, 'g.', markersize=1, linewidth=1)  # 绿线标出钨丝位置
            pylab.plot(x_right, y, 'g.', markersize=1, linewidth=1)
            pylab.show()
        return profile_left, profile_right

    def find_x_off(self, inds):  # 原始左右两侧像素值曲线，钨丝位置的索引
        offsets = [-1, 0, 1]
        off_left = offsets[
            scipy.stats.mode(self.ROI_A[0][:, min(inds):max(inds) + 1].argmax(axis=0))[0][0]]  # 钨丝所在位置，offsets的众数
        off_right = offsets[scipy.stats.mode(self.ROI_A[1][:, min(inds):max(inds) + 1].argmax(axis=0))[0][0]]
        # print(off_left,off_right)
        return off_left, off_right

    def locate_profile_sagittal(self, displayImage=False):  # 由左右两边的像素值曲线组合定位到钨丝
        pa = self.phantom.image
        d_wire = int(self.dis_sagi * self.number_samples / self.High)  # 每条钨丝间的距离，以取样点数为单位
        profile_left, profile_right = self.get_profile_coro_sagi(displayImage=False)
        profile_left = profile_left[2000:self.number_samples - 2000]
        profile_right = profile_right[2000:self.number_samples - 2000]
        p_l, pl_max = self.smooth_curve(profile_left)
        p_r, pr_max = self.smooth_curve(profile_right)
        tl = (p_l.max() - p_l.min()) * 0.9 + p_l.min()
        tr = (p_r.max() - p_r.min()) * 0.9 + p_r.min()
        indicesl = numpy.where(numpy.logical_and(pl_max == p_l, p_l > tl))[0]
        indicesr = numpy.where(numpy.logical_and(pr_max == p_r, p_r > tr))[0]
        print(indicesl)
        print(indicesr)
        if len(indicesl) > 3 or len(indicesr) > 3:  # 很可能是中心定位有误，使报错'该像素值曲线没有定位到钨丝'
            pl_max = numpy.ones(len(profile_left))
            pr_max = numpy.ones(len(profile_right))
        else:  # 对齐左右的像素值曲线
            indl_c = indicesl[int(len(indicesl) / 2)]
            indr_c = indicesr[int(len(indicesr) / 2)]
            if indl_c > indr_c:
                d = (indl_c - indr_c) % d_wire  # 取余，找出要平移的距离
                pr_max = scipy.ndimage.shift(pr_max, int(d), mode='wrap')
                self.move_right = int(d)
            else:
                d = (indr_c - indl_c) % d_wire  # 取余，找出要平移的距离
                pl_max = scipy.ndimage.shift(pl_max, int(d), mode='wrap')
                self.move_left = int(d)
        profile = numpy.maximum(pl_max, pr_max)
        # profile = pl_max+pr_max
        if displayImage:
            pylab.figure()
            pylab.plot(profile_left, 'g')  # 像素值曲线
            pylab.plot(profile_right, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
            pylab.plot(pl_max, 'g')  # 像素值曲线
            pylab.plot(pr_max, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
        return profile

    def get_thickness_sagittal(self, profile):  ######冠状面的层厚计算 Coronal plane
        DEBUG = False
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        if (profile.max() - profile.mean()) <= 400:  # 设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            print(profile.max() - profile.mean())
            print('该像素值曲线没有定位到钨丝')
            return None

        pro = scipy.ndimage.filters.gaussian_filter1d(profile, 50)
        pro[numpy.where(pro < pro.mean())] = pro.mean()  # 将像素值曲线低于均值的部分设为均值
        if self.sliceThickness < 1:  # 设置不同的阈值
            k = 0.996
        elif self.sliceThickness < 2:
            k = 0.97
        elif self.sliceThickness < 3:
            k = 0.8
        else:
            k = 0.65
        threshold = (pro.max() - pro.min()) * k + pro.min()
        inds = numpy.where(pro > threshold)[0]

        lb, nlb = scipy.ndimage.label(pro > threshold)  # nlb看定位到了几条钨丝
        thickness = len(inds) * self.phantom_high / self.number_samples / nlb
        thickness = thickness / tan(self.dip)

        print('层厚测量值：', thickness)
        if DEBUG:
            pylab.plot(pro)
            pylab.plot(numpy.ones(len(pro)) * threshold, 'r')
            pylab.show()

        inds = inds + 2000
        off_left, off_right = self.find_x_off(inds)  # 找到象素值曲线x的偏移量

        YLabel_left = (self.y0 - self.whole + (inds) * self.High / self.number_samples).astype(
            numpy.int) - self.move_right * self.High / self.number_samples
        YLabel_right = (2 * (self.y0 - self.whole) + self.High - YLabel_left).astype(
            numpy.int) + self.move_right * self.High / self.number_samples
        XLabel_left = numpy.ones(len(YLabel_left)) * (self.x0 - self.interval + off_left)
        XLabel_right = numpy.ones(len(YLabel_right)) * (self.x0 + self.interval + off_right)
        YLabel = numpy.hstack((YLabel_left, YLabel_right))
        XLabel = numpy.hstack((XLabel_left, XLabel_right))
        self.LabelPos = [XLabel, YLabel]
        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()

        return thickness  # 层厚测量值

    def locate_profile_coronal(self, displayImage=False):  # 由左右两边的像素值曲线组合定位到钨丝
        pa = self.phantom.image
        profile_left, profile_right = self.get_profile_coro_sagi()
        p_l, pl_max = self.smooth_curve(profile_left)
        p_r, pr_max = self.smooth_curve(profile_right)
        tl = (p_l.max() - p_l.min()) * 0.9 + p_l.min()
        tr = (p_r.max() - p_r.min()) * 0.9 + p_r.min()
        indicesl = numpy.where(numpy.logical_and(pl_max == p_l, p_l > tl))[0]
        indicesr = numpy.where(numpy.logical_and(pr_max == p_r, p_r > tr))[0]
        print(indicesl)
        print(indicesr)
        if len(indicesl) > 1 and len(indicesr) > 1:
            indicesl = [numpy.mean(numpy.where(pl_max == max(pl_max)))]
            indicesr = [numpy.mean(numpy.where(pr_max == max(pr_max)))]
        if len(indicesl) > 1 and len(indicesr) == 1:
            d = abs(indicesl - indicesr[0])
            d_ind = numpy.where(d == min(d))  # 本该只有一个峰值，若其中一条像素值曲线有不止一个峰值，找离另一条曲线峰值距离最近的峰值索引
            indicesl = [indicesl[d_ind]]
        if len(indicesl) == 1 and len(indicesr) > 1:
            d = abs(indicesr - indicesl[0])
            d_ind = numpy.where(d == min(d))  # 距离最近的索引
            indicesr = [indicesr[d_ind]]  # 使len(indicesl) == 1 且 len(indicesr) == 1

        if len(indicesl) == 1 and len(indicesr) == 1:
            pl_max = scipy.ndimage.shift(pl_max, int(indicesr[0] - indicesl[0]),
                                         mode='wrap')  # 对齐峰值，将像素值曲线向右回环平移len(pro)/2
            self.move_left = int(indicesr[0] - indicesl[0])
        profile = numpy.maximum(pl_max, pr_max)
        # profile = pl_max+pr_max
        if displayImage:
            pylab.figure()
            pylab.plot(profile_left, 'g')  # 像素值曲线
            pylab.plot(profile_right, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
            pylab.plot(pl_max, 'g')  # 像素值曲线
            pylab.plot(pr_max, 'b')  # 像素值曲线
            pylab.plot(profile, 'r')  # 像素值曲线
            pylab.show()
        return profile

    def get_thickness_coronal(self, profile):  ######冠状面的层厚计算 Coronal plane
        DEBUG = False
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        if (profile.max() - profile.mean()) <= 400:  # 设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            print(profile.max() - profile.mean())
            print('该像素值曲线没有定位到钨丝')
            return None

        pro = scipy.ndimage.filters.gaussian_filter1d(profile, 50)
        pro[numpy.where(pro < pro.mean())] = pro.mean()  # 将像素值曲线低于均值的部分设为均值
        if self.sliceThickness < 1:  # 设置不同的阈值
            k = 0.97
        elif self.sliceThickness < 2:
            k = 0.8
        else:
            k = 0.6
        threshold = (pro.max() - pro.min()) * k + pro.min()
        inds = numpy.where(pro > threshold)[0]
        thickness = len(inds) * self.phantom_high / self.number_samples
        thickness = thickness * tan(self.dip)
        print('层厚测量值：', thickness)
        if DEBUG:
            pylab.plot(pro)
            pylab.plot(numpy.ones(len(pro)) * threshold, 'r')
            pylab.show()

        off_left, off_right = self.find_x_off(inds)

        YLabel_left = (self.y0 - self.whole + (inds - self.move_left) * self.High / self.number_samples).astype(
            numpy.int)
        YLabel_right = (2 * (self.y0 - self.whole) + self.High - YLabel_left).astype(
            numpy.int) - self.move_left * self.High / self.number_samples
        XLabel_left = numpy.ones(len(YLabel_left)) * (self.x0 - self.interval + off_left)
        XLabel_right = numpy.ones(len(YLabel_right)) * (self.x0 + self.interval + off_right)
        YLabel = numpy.hstack((YLabel_left, YLabel_right))
        XLabel = numpy.hstack((XLabel_left, XLabel_right))
        self.LabelPos = [XLabel, YLabel]

        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()
        return thickness  # 层厚测量值

    def get_profile_transverse(self, displayImage=False):
        pa = self.phantom.dicom.pixel_array  # 模体图像像素矩阵
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        r = int(75.6 / spacing / 2)  # 测层厚的边到中心的距离7.5mm
        xr, yr = int(self.phantom.center_x), int(self.phantom.center_y)  # 模体中心

        rod_R = int(numpy.array(find_rod_locations(pa, spacing, returnR=True)).mean() - 7.5 / spacing)

        ##        print(xr,yr,r,rod_R)
        roi = roi_generator(pa.shape, xr, yr, rod_R)
        roi = pa * roi
        offsets = list(range(-4, 8, 2))  # [-1,0,1]
        ##        profile = []

        pf_left = []#没用到，建议删掉
        pf_right = []
        pf_up = []
        pf_down = []

        # 左边
        y = [list(range(yr - r + off, yr + r + off, 1)) for off in offsets]
        x = [[xr - r + off for m in range(2 * r)] for off in offsets]
        pf_left = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                                 order=3, mode='wrap') for i in range(len(x))]

        pf_left = numpy.array(pf_left)
        off = offsets[scipy.stats.mode(pf_left.argmax(axis=0))[0][0]]
        pf_left = pf_left.max(axis=0)

        ##        print(pf_left)

        thick_left = (self.get_thickness_transverse(pf_left))[0]
        inds_left_y = [i + yr - r + off for i in (self.get_thickness_transverse(pf_left)[1])]
        inds_left_x = [xr - r + off for m in range(len(inds_left_y))]

        # 右边

        y = [list(range(yr - r + off, yr + r + off, 1)) for off in offsets]
        x = [[xr + r + off for m in range(2 * r)] for off in offsets]
        pf_right = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                                  order=3, mode='wrap') for i in range(len(x))]

        # 存储边上的像素值，右边
        pf_right = numpy.array(pf_right)
        off = offsets[scipy.stats.mode(pf_right.argmax(axis=0))[0][0]]
        ##        print(pf_right.argmax(axis=0))
        pf_right = pf_right.max(axis=0)

        ##        print(len(pf_right),off,len(y))
        thick_right = (self.get_thickness_transverse(pf_right))[0]

        inds_right_y = [i + yr - r + off for i in (self.get_thickness_transverse(pf_right)[1])]
        inds_right_x = [xr + r + off for m in range(len(inds_right_y))]

        # 上边
        x = [list(range(xr - r + off, xr + r + off, 1)) for off in offsets]
        y = [[yr - r + off for m in range(2 * r)] for off in offsets]
        pf_up = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                               order=3, mode='wrap') for i in range(len(x))]

        pf_up = numpy.array(pf_up)
        off = offsets[scipy.stats.mode(pf_up.argmax(axis=0))[0][0]]
        pf_up = pf_up.max(axis=0)

        thick_up = (self.get_thickness_transverse(pf_up))[0]

        inds_up_x = [i + xr - r + off for i in (self.get_thickness_transverse(pf_up)[1])]
        inds_up_y = [yr - r + off for m in range(len(inds_up_x))]
        # 下边
        x = [list(range(xr - r + off, xr + r + off, 1)) for off in offsets]
        y = [[yr + r + off for m in range(2 * r)] for off in offsets]
        pf_down = [scipy.ndimage.map_coordinates(roi, numpy.vstack((y[i], x[i])),  # 将pa中坐标为(y,x)的像素提取出来
                                                 order=3, mode='wrap') for i in range(len(x))]

        pf_down = numpy.array(pf_down)
        off = offsets[scipy.stats.mode(pf_down.argmax(axis=0))[0][0]]
        pf_down = pf_down.max(axis=0)

        thick_down = (self.get_thickness_transverse(pf_down))[0]

        inds_down_x = [i + xr - r + off for i in (self.get_thickness_transverse(pf_down)[1])]
        inds_down_y = [yr + r + off for m in range(len(inds_down_x))]

        indsx = list(inds_left_x) + list(inds_right_x) + list(inds_up_x) + list(inds_down_x)
        indsy = list(inds_left_y) + list(inds_right_y) + list(inds_up_y) + list(inds_down_y)

        self.LabelPos = [indsx, indsy]
        ##        print(self.LabelPos,len(indsx),len(indsy))
        thickness = ((thick_left + thick_right) * 2 * 0.75 + (thick_up + thick_down) / 2 * 0.83) / 4
        if displayImage:  # 圆环上取样点的像素值曲线图
            pylab.plot(pf_left, 'b')
            pylab.plot(pf_right, 'b')
            pylab.plot(pf_up, 'r')
            pylab.plot(pf_down, 'r')
            pylab.show()
        ##            print ("the shape of the profile:", profile.shape)
        ##            for i in range(profile.shape[0]):
        ##                pylab.plot(profile[i])
        ##            pylab.show()
        print("thickness = ")
        print(thickness)
        return thickness

    def get_thickness_transverse(self, profile):
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        pro = scipy.ndimage.median_filter(profile, 5)
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        ##        print maxv-mean
        if (maxv - mean) <= 300:  # 20200509,100     设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            ##            print (maxv-mean)
            ##            print ('该像素值曲线没有定位到钨丝')
            return [0, []]

        pro[numpy.where(pro < pro.mean())] = pro.mean()
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        threshold = (mean + maxv) / 2
        inds0 = numpy.where(pro > threshold)[0]
        pro = scipy.ndimage.shift(pro, int(len(pro) / 2 - numpy.argmax(pro)), mode='wrap')
        inds = numpy.where(pro > threshold)[0]
        ##        print(inds[-1],inds[0],len(inds))
        length = (len(inds) - 1) * spacing

        ##        print(length*scipy.tan(26.57/180*scipy.pi),length/scipy.tan(26.57/180*scipy.pi))
        ##        pylab.plot(pro)
        ##        pylab.plot([0,len(pro)],[threshold,threshold])
        ##        pylab.show()
        return length, inds0

    # 将极坐标转换为笛卡尔坐标
    # angel：待转化的极坐标的角度数组；rho：极坐标半径；center_coor：极坐标极点；
    def angle2coor(self, angle, rho, center_coor, as_index=False):
        # convert polar angle to cartesian coordinates
        #   This controls where the starting point is in the CT phantom image
        #   currently, the zero angle is located at the right-most point of a circle
        yc, xc = center_coor  # 极点的直角坐标
        x = xc + rho * numpy.cos(angle)  # 将数组angle转化为直角坐标，0角位于圆的最右边
        y = yc + rho * numpy.sin(angle)
        # because the coordinates may be used to do subpixel sampling
        # here, the coordinates can be float values
        # but you have an option to convert them to image pixel indices
        if as_index:  # 若坐标被用作图像像素索引
            return (scipy.uint16(scipy.round_(y, 0)),  # 将浮点型坐标值转化为整型
                    scipy.uint16(scipy.round_(x, 0)))
        else:
            return (y, x)  # 数组angle的直角坐标

    # 得到螺旋珠所在的圆环上的点的角度和像素值曲线
    def get_profile(self, displayImage=False):
        # rename variables for convenience
        pa = self.phantom.dicom.pixel_array  # 模体图像像素矩阵
        rad = self.rou  # 以像素个数为单位的圆环半径
        xr, yr = self.phantom.center_x, self.phantom.center_y  # 模体中心
        # need a profile to get degMax (position of the brightest bead)
        thetaPlt = numpy.linspace(0, 2 * scipy.pi, self.number_samples)  # 在0到2pi内均匀取25000个数值
        profile = []  # 用来存储像素值的空白数组
        # in the beginning, the precision of the model is not good
        # therefore, used to have average over different radii
        # but the current phantom is good enough
        # therefore, no need to do averaging any more

        # to romve possible inaccuracy in bead-mounting
        #    and to romve the air gaps in the profile
        #    using the maximum to replace the mean 06/19/2018
        offsets = [-1, 0, 1]  # 圆环半径偏差值
        for off in offsets:
            y, x = self.angle2coor(thetaPlt, rad + off, (yr, xr), as_index=False)  # 将取样点的极坐标转化为直角坐标
            # scipy provides below function to do interpolation
            # using spline interpolation with the order of 3
            # when an edge is encountered, the "wrap" mode is used
            pf = scipy.ndimage.map_coordinates(pa, numpy.vstack((y, x)),  # 将pa中坐标为(y,x)的像素提取出来
                                               order=3, mode='wrap')
            profile.append(pf)  # 存储圆环上的像素值
        profile = numpy.array(profile)
        if displayImage:  # 圆环上取样点的像素值曲线图
            print("the shape of the profile:", profile.shape)
            for i in range(profile.shape[0]):
                pylab.plot(profile[i])
            pylab.show()
        ##        profile = profile.mean(axis=0)
        profile = profile.max(axis=0)  # 综合三条像素值曲线得到一条，同一位置选择最大像素值
        ##        print "profile shape =", profile.shape

        if displayImage:
            pylab.figure()
            pylab.imshow(pa, interpolation="nearest", cmap='gray')  # 原始模体图像
            pylab.plot(x, y, 'g.', markersize=1, linewidth=1)  # 绿线标出钨珠所在圆环
            pylab.show()
            # print "profile mean = %s"%(profile.mean())
            pylab.plot(profile)  # scipy.ndimage.filters.median_filter(profile,5))  #角度-像素值曲线
            pylab.show()
        return {'theta': thetaPlt, 'profile': profile}

    # 由钨珠分布的角度大小计算层厚测量值
    def get_lthickness(self, profile, bc=0.625):
        DEBUG = False
        pitch = self.pitch  # 螺距
        spacing = self.phantom.dicom.PixelSpacing[0]  # 像素间距（mm）
        # 针对不同的标称值设定中值滤波器的参数和阈值的相对位置
        if bc <= 0.55:
            a = 2.2
            k = 55 / 98.  # 141/223.
        elif bc <= 0.625:
            a = 3.08
            k = 127 / 223.
        elif bc <= 1.1:
            a = 30.8
            k = 55 / 108.
        elif bc <= 1.25:
            a = 66
            k = 55 / 119.
        elif bc <= 2.2:
            a = 118.8
            k = 55 / 112.
        elif bc <= 5:
            a = 132
            k = 55 / 108.
        elif bc <= 5.5:
            a = 162.8
            k = 55 / 113  # 55/113.
        else:
            a = 176
            k = 55 / 115.

        pro = scipy.ndimage.median_filter(profile['profile'], int(a / spacing))  # 中值滤波以平滑profile，参数设置：滤波窗口的像素长（邻域）
        mean = pro.mean()  # pro为长度为25000的1维数组
        maxv = pro.max()
        ##        print maxv-mean
        if (maxv - mean) <= 60:  # 20200509,100     设定阈值，判断像素值曲线是否找到了钨丝所在的位置，没有找到，返回None，找到了才继续计算
            print(maxv - mean)
            print('该像素值曲线没有定位到钨丝')
            return None

        threshold = k * (mean + maxv)  # 像素值阈值，k
        ##        threshold =  (pro.mean()*2 + pro.max())/3
        inds = numpy.where(pro > threshold)[0]  # 筛选出大于阈值的像素，返回其索引
        #
        # outer_edge = scipy.logical_xor(binary_dilation(outer, [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        #                                # xor是异或，不相同则为1。将outer膨胀，与原来的的outer异或，得到的结果就是边界
        #                                outer)
        spread = pro > threshold  # condition
        lb, nlb = scipy.ndimage.label(spread)

        hist, le = scipy.histogram(lb, bins=list(range(1, nlb + 2)))
        i = scipy.argmax(hist) + 1
        # print(numpy.where(lb ==1 ),i,inds,'钨丝范围连通域数目:', nlb)
        if nlb > 1:
            inds = numpy.where(lb == 1)[0]
        inds_theta = inds * scipy.pi * 2 / self.number_samples  # 钨丝所在角度值范围

        flag_move1 = 0
        if inds[0] == 0:  # 若像素值曲线刚好是从钨珠所在位置开始截取的
            pro = scipy.ndimage.shift(pro, int(len(pro) / 2), mode='wrap')  # 将像素值曲线向右回环平移len(pro)/2
            inds = numpy.where(pro > threshold)[0]  # 新的索引
            flag_move1 = 1
        tht = profile['theta']

        span = (tht[inds[-1]] - tht[inds[0]]) / scipy.pi * (pitch / 2)  # 由钨珠分布的角度和螺距计算层厚

        thickness = span
        # print ("高斯拟合前钨珠角度法所得层厚测量值 = %s"%span)
        #        FUBUMOTI = 1
        flag_move2 = 0
        if abs(numpy.argmax(pro) - len(pro) / 2) > len(pro) / 5:  # 尽量平移像素值曲线，使峰值不太偏
            distance = int(len(pro) / 2 - numpy.argmax(pro))
            pro = scipy.ndimage.shift(pro, int(len(pro) / 2 - numpy.argmax(pro)), mode='wrap')
            inds = numpy.where(pro > threshold)[0]  # 新的索引
            flag_move2 = 1
        pro[numpy.where(pro < pro.mean())] = pro.mean()  # 将像素值曲线低于均值的部分设为均值
        # print (pro.mean())

        condition = (pro.max() - pro.min()) * 0.3 + pro.min()
        spread = pro > condition
        lb, nlb = scipy.ndimage.label(spread)
        print('钨丝范围连通域数目:', nlb)
        if nlb > 1:
            print('钨丝范围连通域数目>1')
            # return None

        if DEBUG:
            pylab.plot(pro, 'b')  # 原像素值曲线
            # pylab.plot(pro2,'g')  #高斯拟合的像素值曲线
            ##            pylab.plot(profile['profile'])
            # pylab.plot(numpy.ones(len(pro))*mean)  #用直线标示原像素值曲线的均值
            pylab.plot(numpy.ones(len(pro)) * condition, 'y')  # 最大值
            pylab.plot(numpy.ones(len(pro)) * threshold, 'r')  # 像素阈值
            pylab.show()

        xr, yr = self.phantom.center_x, self.phantom.center_y  # 模体中心
        inds_yx = self.angle2coor(inds_theta, self.rou, (yr, xr), as_index=False)  # 钨丝位置
        self.LabelPos = [inds_yx[1], inds_yx[0]]  # x,y

        if DEBUG:
            pylab.figure()
            pylab.imshow(self.phantom.image, interpolation="nearest", cmap='gray')
            pylab.plot(self.LabelPos[0], self.LabelPos[1], 'r.', markersize=1, linewidth=1)  # 钨丝位置
            pylab.show()

        return thickness  # 层厚测量值


SECTION_TYPE = [0, 1]
GEOMETRY = {
    0: [  # diameters
        161,  # mm, the outer diameter
        110,  # mm, the diameter of the circle where the 8 linearity rods locate
        15,  # mm, the linearity rod diameter
        15,  # mm, the MTF wire rod diameter
        3,  # mm, the diameter of the 4 geometric distortion holes

        # distances
        90,  # mm, pitch of the spiral beads
        32,  # mm, the length of the hole modules
        10,  # mm, the depth of the hole modules
        0,  # mm, the distance from the center to the geometric distrotion holes, UNKNOWN
        30,  # mm, the distance from the center to the MTF wire rod center
    ],
    1: [  #
        161,  # 145,
        100,
        15,
        15,
        3,
        #
        70,  # mm, pitch of the spiral beads
        32,  # ?
        10,  # ?
        0,  # UNKNOWN
        25,
    ],
    2: [  #
        161,  # 113, # ?
        90,
        12,
        12,
        3,
        #
        60,  # mm, planned pitch of the spiral beads
        32,  # ?
        10,  # ?
        0,  # UNKNOWN
        0,  # UNKNOWN
    ]
}


class CT_phantom:
    """
    The structure of phantom.
    In this design, there are two phantom sections.
    One is a water phantom, including a cylindrical container with water
        inside and probably shells outside the container
    the other is a comprehensive phantom, including several components
        bead spiral for thickness, square holes for spatial resolution,
        four small cylindrical holes for geometrical distortion,
        eight cylindrical rods for CT number linearity,
        and a tungen wire for spatial resolution
    This class is to identify which phantom section the image is
    and to locate each component in the phantom section
    """

    def __init__(self, dcm_img):
        if type(dcm_img) in [type("string"), type(u"string")]:
            # assume this is a dicom file
            try:
                dcm = dicom.read_file(dcm_img, force=True)
            except:
                print("Not a dicom file: %s" % dcm_img)
                # return False
        else:
            dcm = dcm_img
        self.dicom = dcm
        self.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

        self.section_type = self.get_section_type()

        re = find_CT_phantom_outer_edge(self.image, dcm.PixelSpacing[0], return_coors=True)  # 定位模体外边缘
        # re = [1,2,3,4,5]
        self.center_x = re[0]  # 模体中心坐标
        self.center_y = re[1]
        self.outer_radius = re[2]  # 模体外边缘半径
        self.outer_coor_xs = re[3]  # 外边界坐标
        self.outer_coor_ys = re[4]

        # find the structure
        if self.section_type == SECTION_TYPE[1]:
            self.determine_size()

    def get_section_type(self):
        """
        determine whether the phantom is the water or the comprehensive section

        since the water section has simpler structure, the edge pixels are less
        therefore, the number of edge pixels is used to tell difference
        """
        edges = canny(self.image, sigma=2.0,
                      low_threshold=50, high_threshold=100)

        if edges.sum() < 7500:
            return SECTION_TYPE[0]
        else:
            return SECTION_TYPE[1]

    def determine_size(self):
        self.find_MTF_wire_rod()
        # self.find_rod_locations()

    def find_MTF_wire_rod(self):
        """
        the design of phantom can be characterized by the distance between
        the phantom center and the MTF wire rod center
        """
        DEBUG = False
        re = [self.center_x, self.center_y, self.outer_radius,
              self.outer_coor_xs, self.outer_coor_ys]
        xc, yc, r, xe, ye = re

        h, w = self.dicom.pixel_array.shape
        mask = scipy.mgrid[0:h, 0:w]
        # detect in this region to see where the wire is
        detection_dist = 40  # mm to cover the MTF wire rod
        detection_dist /= self.dicom.PixelSpacing[0]
        dist_map = numpy.hypot((mask[0] - yc), (mask[1] - xc))

        detection_zone = dist_map < detection_dist

        # to determine how to smooth the image
        #   with a high SNR, smaller kernel may be used
        std = numpy.std(self.image[numpy.where(detection_zone)])
        try:
            kernel = self.dicom.ConvolutionKernel
        except:
            kernel = None

        if kernel == "BONE" or std > 40:
            sigmaV = 3
        else:
            sigmaV = 1

        # print "using sigma = %s, std = %s"%(sigmaV, self.image[scipy.where(detection_zone)].std())
        edge = canny(self.image, sigma=sigmaV,
                     low_threshold=10, high_threshold=100)
        edge *= detection_zone

        # to find the largest region
        #   which can be assumed to be the wire rod
        lb, nlb = label(edge == 0)
        # print nlb
        if nlb == 1:
            # could not detect the MTF wire rod
            print("Could not detect the MTF wire rod!")
            return
        hist, le = numpy.histogram(lb, bins=range(2, nlb + 2))
        ind = numpy.argsort(hist)[-1] + 2
        rod = lb == ind

        # the distance between the center of the MTF wire and the center of the phantom
        rodyc, rodxc = [numpy.mean(e) for e in numpy.where(rod)]
        dist_cc = numpy.hypot(xc - rodxc, yc - rodyc)
        dist_cc_mm = dist_cc * self.dicom.PixelSpacing[0]
        # print "distance between the MTF rod and the center:", dist_cc_mm

        if DEBUG:
            import pylab
            # pylab.imshow(lb)
            # pylab.show()
            pylab.imshow(rod)
            pylab.show()

        ind = -1
        err = 1000.
        for k in GEOMETRY.keys():
            abs_err = abs(dist_cc_mm - GEOMETRY[k][-1])
            if err > abs_err:
                ind = k
                err = abs_err
        ##        print "geometry type:", ind
        self.geometry = GEOMETRY[ind]


####################################################################################
if __name__ == "__main__":
    import os
    import pydicom as dicom

    # fname = "D:/motituxiang/motituxiang/横断-体检中心0515\Z408"
    # ##    fname = "D:/motituxiang/motituxiang/冠状和矢状-3\Z1002"
    # dcm= dicom.read_file(fname)
    # phantom = CT_phantom(dcm)
    # spiralbeads = SpiralBeads(phantom, interval = 72, dip = 26.56)#diameter=75, pitch=90,number_beads=180
    # profile1 = spiralbeads.get_profile_transverse(displayImage=True)
    # fname = u"D:\\CT-D-phantom\\矢状面\\Z354"  # Z589,615,467
    # fname = u"D:\\医学模体图像\\TEST1.5+test1.5-1.5\\TEST1.5+test1.5-1.5\\Z13"
    fname = u"D:\\医学模体图像\\儿童模数据12月\\test2.3系列-1.8-1.8\\Z85"
    dcm = dicom.read_file(fname)
    phantom = CT_phantom(dcm)
    # spiralbeads = SpiralBeads(phantom, interval = 72, dip = 26.56)
    spiralbeads = SpiralBeads(phantom, diameter=75, pitch=90, number_beads=180)

    profile = spiralbeads.get_profile(displayImage=True)
    thickness = spiralbeads.get_lthickness(profile, bc=dcm.SliceThickness)
    # profile = spiralbeads.locate_profile_coronal(displayImage = True)  #sagittal
    # thickness = spiralbeads.get_thickness_coronal(profile)

    # '''
    # pname = u"D:\\CT-D-phantom\\矢状面"
    #
    # files = sorted(os.listdir(pname),key=alphanum_key)
    #
    # zonghe=[]
    # biaocheng=[]
    # slice=[]
    # WuCha=[]
    # a=0
    # wb=xlwt.Workbook()
    # sh=wb.add_sheet('zonghe')
    #
    # for f in files[0:20]:
    #     fname = os.path.join(pname, f)
    #     print (fname)
    #     dcm = dicom.read_file(fname)
    #     phantom = CT_phantom(dcm)
    #
    #     spiralbeads = SpiralBeads(phantom, interval = 72, dip = 26.56)
    #     profile = spiralbeads.locate_profile_sagittal(displayImage=False)
    #     #profile_left,profile_right = spiralbeads.get_profile_coro_sagi(displayImage=False)
    #     #profile_right = profile_right[2000:spiralbeads.number_samples-2000]
    #     thickness = spiralbeads.get_thickness_sagittal(profile)
    #     if thickness is None:
    #         print ("cannot estimate the slice thickness!")
    #     else:
    #         print ("The measured slice thickness is %f"%thickness)
    #
    #         wuchazhi=(thickness-dcm.SliceThickness)*100/dcm.SliceThickness
    #         zonghe.append(str(fname).replace('矢状',''))
    #         biaocheng.append(str(dcm.SliceThickness))
    #         slice.append(str(thickness))
    #         WuCha.append(str(wuchazhi))
    #         a=a+1
    #         sh.write(a,0,zonghe[-1:])
    #         sh.write(a,1,biaocheng[-1:])
    #         sh.write(a,2,slice[-1:])
    #         sh.write(a,3,WuCha[-1:])
    #
    # sh.write(0,0,u'图像名称')
    # sh.write(0,1,u'层厚标称值')
    # sh.write(0,2,u'层厚测量值')
    # sh.write(0,3,u'测量误差%')
    #wb.save('冠状面-左右各偏移两个像素.xls')

    #pylab.boxplot(float(slice))
    #pylab.show()
    # '''
