# -*- coding: utf-8 -*-
"""
Created on Wed May 03 10:49:52 2017

@author: xma_m
"""
import os, sys, scipy, pylab
import numpy as np
from utils.util import find_CT_phantom_outer_edge, read_dicom_file, find_edge_new

currentfilefolder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(currentfilefolder, 'Util')))


def freq_at_mtf_perc(feq, mtf, perc):
    """
    find the frequencies at given %mtf
    the first two parameters are the mtf plot
    more specifically, the first is the frequency list
                       the second is the mtf at the frequencies
    the last parameter is a %mtf whose frequency needs to be calculated
    找到给定mtf对应的频率
    第一个参数是频率列表，第二个参数是频率上的mtf
    最后一个参数是需要计算其频率的%mtf
    """
    # NOTE: mtf here is a function with max of 1      这里的mtf是一个最大值为1的函数
    if mtf.min() > perc:  # mtf值降不到给定值的情况
        return -1

    # in case there is a point satisfying the percentage
    if scipy.any(mtf == perc):  #
        return feq[scipy.where(mtf == perc)]
    # try to find the two points where the targeting percentage is in between
    inds1 = scipy.where(mtf >= perc)[0].tolist()
    ind2 = scipy.where(mtf < perc)[0][0]
    # to get rid of the indices that come from the wiggle at the end of the spectrum
    #  in some cases, the MTF curve raises up at the higher frequency
    ind1 = inds1.pop(-1)
    while ind1 != len(inds1):
        ind1 = inds1.pop(-1)
    # print "the indices enclosing the target frequecy:", ind1,ind2
    f1, f2, m1, m2 = feq[ind1], feq[ind2], mtf[ind1], mtf[ind2]
    # print "the frequencies and the MTF's: "f1, f2, m1, m2
    return f2 - (f2 - f1) * (perc - m2) / (m1 - m2)


def FFT_2D(psf, sz=0, fftshift=True):
    """
    perform 2D Fourier transform (FT) for a 2D PSF function
    with or without zero-padding
    when sz is less than the size of the psf, do not do zero-padding
    else do zero-padding
    if fftshift is set, do fftshift so that the zero-frequency is at the center
对二维PSF函数执行二维傅里叶变换（FT）
可选择带或不带零填充
当sz小于psf的大小时，不要进行零填充；否则做零填充
如果设置了fftshift，则执行fftshift以使零频率位于中心
    """
    h, w = psf.shape
    if h < w:
        d = h
    else:
        d = w

    if sz > d:
        zero_padded = np.zeros((sz, sz))
        zero_padded[int((sz - h) / 2): int((sz + h) / 2),
        int((sz - w) / 2): int((sz + w) / 2)] = psf
    else:
        zero_padded = psf

    # mtf_2d = np.fft.fftshift(np.fft.fftn(PSF_ROI))
    # mtf_2d = np.fft.fftshift(np.fft.fftn(zero_padded))
    mtf_2d = np.fft.fftn(zero_padded)
    if fftshift:
        mtf_2d = np.fft.fftshift(mtf_2d)
    return scipy.absolute(mtf_2d)


def fft2DRadProf(ft, d=1.):
    coord = np.fft.fftshift(np.fft.fftfreq(ft.shape[0], d))
    image = np.fft.fftshift(ft)

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    center = np.squeeze(np.array([x[0, coord == 0.0], y[coord == 0.0, 0]]))
    r = np.hypot(x - center[0], y - center[1])  # compute the distance of every point to (0,0)
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]  # list of distance in a sorted manner
    i_sorted = image.flat[ind]  # list of "frequency"(?) sorted according to distance

    # Get the integer part of the radii (bin size = 1)
    r_int = np.rint(r_sorted).astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of points in each radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    cl = np.rint(np.sum(coord >= 0))  # so that coord and radial_prof have the same number of points??????

    return coord[coord >= 0], radial_prof[:int(cl)]


def roi_generator(imageshape, xc, yc, radius):  # 用来产生roi区域
    img = scipy.ones(imageshape)
    img[yc, xc] = 0
    newimage = scipy.ndimage.distance_transform_edt(img)  # distance_transform_edt函数用来求解非零像素到零像素点的最短距离
    roi = newimage < radius  # 距离小于掩膜半径的为1
    return roi


class TungstenBead:  # 钨珠类
    def __init__(self, phantomimage, roiwidth=32, windowwidth=11, orientation_flag=2, **kwargs):
        self.ISPOS = True
        dcm, dpa, ds = read_dicom_file(phantomimage)
        self.dcm = dcm  # dcm是dicom对象
        self.dcm_shape = dcm.pixel_array.shape  # 一般是512*512的阵列
        self.orientation = orientation_flag
        self.__phantomimage = dpa
        self.__pixelspacing = ds
        self.__roiwidth = roiwidth / dcm.PixelSpacing[0] * 0.44
        self.__windowwidth = windowwidth
        self.__roi = None
        self.resolution = None
        self.mtf = None
        self.freq = None
        self.pos = self.__InitPosition(**kwargs)  # **kwargs是钨珠中心位置坐标
        self.__backgroundvalue = self.__EstimateBackgroundValue()  # 计算ROI内模体背景的CT值
        self.psf = self.__GetPSF()  # 得到roi内的二维点扩展函数并进行加窗处理
        self.__CalculateMTF(**kwargs)  # 计算frequency-mtf曲线和mtf为0.1和0.5时对应的高对比分辨力值（LP/cm）

    def __InitPosition(self, **kwargs):  # 返回钨珠位置的数组
        #       r = self.__roiwidth / 2
        r = int(self.__roiwidth / 2 + 0.5)
        flag = self.orientation
        if 'bead_position' in kwargs:
            #            beadpositionx, beadpositiony = scipy.where(self.__phantomimage == self.__phantomimage.max())
            beadpositionx, beadpositiony = kwargs['bead_position']
        elif kwargs:
            fh = pylab.figure(dpi=150, figsize=(10, 10))
            pylab.imshow(self.__phantomimage, cmap='gray', interpolation='nearest')
            pylab.title('Pinpoint the Tungsten Bead/Line')
            pos = pylab.ginput(1)[0]
            pylab.close(fh)
            xc, yc = pos
            self.__roi = scipy.array(self.__phantomimage[yc - r: yc + r + 1, xc - r: xc + r + 1], dtype=scipy.float32)
            [beadpositiony, beadpositionx] = scipy.where(self.__roi == self.__roi.max())
            beadpositiony, beadpositionx = beadpositiony + yc - r, beadpositionx + xc - r
        else:
            # 自动计算
            print(flag)
            if flag == 2:  # 横截面
                xc, yc, r_outer = find_CT_phantom_outer_edge(self.__phantomimage, self.dcm.PixelSpacing[0],
                                                             return_coors=False)  # 返回综合模体的圆心坐标、半径、是否为腹模
                print(xc, yc)
                xc_new = int(round(xc))  # -19/self.__pixelspacing
                yc_new = int(round(yc))  # int(round(yc+24/self.__pixelspacing))
            else:
                xc, yc, xs, ys, x1, x2, y1, y2 = find_edge_new(self.__phantomimage,
                                                               self.dcm.PixelSpacing[0])  # 返回综合模体的底边中心坐标、边缘坐标
                print(xc, yc)
                if flag == 0:  # 矢状位

                    xc_new = int(round(xc - 17 / self.__pixelspacing))
                    yc_new = int(round(yc - 158 / self.__pixelspacing))
                else:  # 冠状位elif flag == 1

                    xc_new = int(round(xc + 17.5 / self.__pixelspacing))
                    yc_new = int(round(yc - 125 / self.__pixelspacing))

            roi = roi_generator(self.dcm_shape, xc_new, yc_new,
                                int(28 / self.__pixelspacing))  # 70,20200509         #参数分别是图像shape，圆心坐标，掩膜半径
            ROI = roi * self.dcm.pixel_array  # 掩膜与图像相乘得到ROI区域
            ##            import pylab
            ##            pylab.imshow(ROI,cmap = "gray")
            ##            pylab.show()
            if ROI.max() * self.dcm.RescaleSlope + self.dcm.RescaleIntercept < 200:
                self.ISPOS = False

            [beadpositiony, beadpositionx] = scipy.where(ROI == ROI.max())  # ROI中CT值最大的是钨珠的位置
            beadpositiony = beadpositiony[0]
            beadpositionx = beadpositionx[0]
            # print(beadpositiony-r , beadpositiony+r+1, beadpositionx-r ,beadpositionx+r+1,yc,xc)

        self.__roi = scipy.array(
            self.__phantomimage[beadpositiony - r: beadpositiony + r + 1, beadpositionx - r: beadpositionx + r + 1],
            dtype=scipy.float32)
        self.beadpositionx = beadpositionx
        self.beadpositiony = beadpositiony
        #       print kwargs
        print(beadpositionx, beadpositiony)
        return scipy.array([beadpositionx, beadpositiony])

    def __EstimateBackgroundValue(self):
        '''
        # displaying the region for background estimation
        a = scipy.zeros(self.__roi.shape)
        a[0:5,:] = 1
        a[-5:,:] = 1
        a[5:-5,0:5] = 1
        a[5:-5,-5:] = 1
        pylab.figure(dpi = 150, figsize = (8, 6))
        imh = pylab.imshow(self.__roi,cmap = 'gray', interpolation = 'nearest')
        pylab.imshow(a, cmap = 'gray', interpolation = 'nearest', alpha = 0.3)
        pylab.title('Area for Background Estimation')
        pylab.colorbar(imh)
        pylab.axis('off')
        '''
        sumforestimation = self.__roi[0:5, :].sum() + self.__roi[-5:, :].sum() + self.__roi[6:-5,
                                                                                 0:5].sum() + self.__roi[6:-5,
                                                                                              -5:].sum()  # 全部CT值之和
        backgroundvalue = float(sumforestimation) / (
                    (self.__roi.shape[0] * 10) + (self.__roi.shape[1] * 10) - 25 * 4)  # 除以提取像素数
        # print 'Estimated Background = ', backgroundvalue

        return backgroundvalue

    def __GetPSF(self):
        """
        print("############", smallwindow.shape)
        print(yc, self.__windowwidth, xc)
        print(fullsizewindow.shape)
        print(fullsizewindow[int(yc-(self.__windowwidth-1)/2) : int(yc+(self.__windowwidth-1)/2+1), \
                int(xc-(self.__windowwidth-1)/2) : int(xc+(self.__windowwidth-1)/2+1)].shape)
                """
        yc, xc = scipy.where(self.__roi == self.__roi.max())

        psf = self.__roi - self.__backgroundvalue
        smallwindow = scipy.matrix(scipy.hanning(self.__windowwidth)).T * scipy.matrix(
            scipy.hanning(self.__windowwidth))
        fullsizewindow = scipy.zeros(psf.shape, dtype=float)
        yc, xc = yc[0], xc[0]

        # print fullsizewindow[0: yc+(self.__windowwidth-1)/2+1, \
        #        xc-(self.__windowwidth-1)/2 : xc+(self.__windowwidth-1)/2+1].shape
        # print fullsizewindow[yc-(self.__windowwidth-1)/2 : yc+(self.__windowwidth-1)/2+1, \
        #        0 : xc+(self.__windowwidth-1)/2+1].shape
        # print xc+(self.__windowwidth-1)/2+1,xc-(self.__windowwidth-1)/2
        print("#######")
        if (yc + (self.__windowwidth - 1) / 2 + 1) > fullsizewindow.shape[0]:
            yc = -(self.__windowwidth - 1) / 2 - 1 + fullsizewindow.shape[0]
        print(yc)
        if yc - (self.__windowwidth - 1) / 2 < 0:
            yc = (self.__windowwidth - 1) / 2
        if xc - (self.__windowwidth - 1) / 2 < 0:
            xc = (self.__windowwidth - 1) / 2
        if (xc + (self.__windowwidth - 1) / 2 + 1) > fullsizewindow.shape[1]:
            xc = -(self.__windowwidth - 1) / 2 - 1 + fullsizewindow.shape[1]

        print(fullsizewindow[int(yc - (self.__windowwidth - 1) / 2): int(yc + (self.__windowwidth - 1) / 2 + 1), \
              int(xc - (self.__windowwidth - 1) / 2): int(xc + (self.__windowwidth - 1) / 2 + 1)].shape)
        fullsizewindow[int(yc - (self.__windowwidth - 1) / 2): int(yc + (self.__windowwidth - 1) / 2 + 1), \
        int(xc - (self.__windowwidth - 1) / 2): int(xc + (self.__windowwidth - 1) / 2 + 1)] = smallwindow
        # mtf2d = FFT_2D(psf*fullsizewindow, sz = 256)
        # '''
        # fig = pylab.figure(dpi = 150, figsize = (10, 10))
        # gs = pylab.matplotlib.gridspec.GridSpec(1, 3)
        # ax0 = pylab.subplot(gs[0])
        # ax1 = pylab.subplot(gs[1])
        # ax2 = pylab.subplot(gs[2])
        # ax0.imshow(psf, cmap = 'jet', interpolation = 'nearest')
        # imh = ax1.imshow(psf * fullsizewindow, cmap = 'jet', interpolation = 'nearest')
        # ax2.imshow(mtf2d, cmap = 'gray', interpolation = 'nearest')
        # ax0.set_title('PSF before applying window')
        # ax1.set_title('PSF after applying window')
        # ax2.set_title('OTF(magnitude)')
        # '''
        # fig.subplots_adjust(top = 0.9, bottom = 0.1)
        # cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
        # fig.colorbar(imh, cax=cbar_ax)

        return psf * fullsizewindow  # 返回加过窗的psf

    def __CalculateMTF(self, **kwargs):  # 计算MTF值
        self.mtf2d = FFT_2D(self.psf, sz=256)  # 对psf傅里叶变换得到mtf值
        freq, mtf = fft2DRadProf(np.fft.fftshift(self.mtf2d), self.__pixelspacing)  # 由具有旋转对称性的二维MTF得到一维MTF
        normalize = lambda mtn: (mtn - mtn.min()) / (mtn[0] - mtn.min())
        mtf = normalize(mtf)  # 将mtf归一化
        fq = freq_at_mtf_perc(freq, mtf, 0.1)
        self.mtf = mtf
        self.freq = freq
        self.resolution10 = fq
        fq = freq_at_mtf_perc(freq, mtf, 0.5)  # 计算给定mtf值对应的高对比分辨力值（LP/mm）
        self.resolution50 = fq


def MTF_show(r10, r50, freq, mtf):  # self,
    # pylab.figure()
    # pylab.imshow(bead.psf, cmap='gray', interpolation='nearest')
    # pylab.figure()
    # pylab.imshow(bead.mtf2d, cmap='gray')
    figurehandle = pylab.figure(dpi=100, figsize=(8, 6))
    pylab.plot(freq, mtf)
    pylab.text(r10, 0.1, '10%% MTF = %.3f' % (r10))
    pylab.text(r50, 0.5, '50%% MTF = %.3f' % (r50))
    pylab.plot(r10, 0.1, '+r')
    pylab.plot(r50, 0.5, '+r')
    pylab.xlabel('frequency/(lp/cm)')
    pylab.ylabel('mtf')
    pylab.title('MTF from Tungsten Wire/Bead')
    # figurehandle.savefig('window width = %d' % windwidth)
    # pylab.ion()
    pylab.show()


if __name__ == "__main__":
    import re


    def alphanum_key(s):
        return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


    # pn="F:\\CT_phantom_pictures"
    # files = os.listdir(pn)                                                                       #打开整个文件夹
    # fnames = [os.path.join(pn, e) for e in files]                                               #遍历全部文件
    # print(fnames)
    if 1:
        for fname in range(1, 25):  # fnames[65:66]:
            # print(fname)
            # fname = fname.replace('/', '\\\\')
            fname = "C:\\Users\\yanhui\\Desktop\\IM0"
            print(fname)
            windwidth = 9  # 对psf加窗处理的窗宽
            bead = TungstenBead(  # dcm.pixel_array, float(dcm.PixelSpacing[0]),
                fname, roiwidth=32,  # roiwidth是感兴趣区域（矩形）的宽度
                windowwidth=windwidth,
                # bead_position=[254,313],                                 #kwargs是空的，是自动计算
            )
            print("bead.resolution10*10", bead.resolution10 * 10)
            print("bead.resolution50*10", bead.resolution50 * 10)
            print("bead.beadpositionx", bead.beadpositionx)
            print("bead.beadpositiony", bead.beadpositiony)
            MTF_show(bead.resolution10 * 10, bead.resolution50 * 10, bead.freq * 10, bead.mtf)
