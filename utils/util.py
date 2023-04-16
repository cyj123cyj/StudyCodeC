# -*- coding: utf-8 -*-
import os
import pydicom as dicom

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
##from scipy.ndimage.filters import gaussian_filter
import math, scipy
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from math import acos, pi
from scipy import optimize
import numpy.linalg as linalg
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import (gaussian_filter,
                           generate_binary_structure, binary_erosion, label)

WATER_DISTANCE = 90  # 冠状面与矢状面测得水模体长度
DISTANCE12 = 12  # 冠状面与矢状面时水模体与综合模体分界的距离
TRANSVERSE = AXIAL = 0
FRONTAL = CORONAL = 1
MEDIAN = SAGITTAL = 2
ALLOWED_PLANES = (AXIAL, CORONAL, SAGITTAL)
ALLOWED_MODALITIES = ('CT', 'MR', 'CR', 'RT')


def read_dicom_file(fname):  # 用于读取dicom图像的函数
    dcm = dicom.read_file(fname)  # 获取dicom对象
    dpa = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept  # 读取单张dicom图像
    ds = dcm.PixelSpacing
    ds = float(ds[0])  # 得到spacing值，其含义是用来表示当前二维图像坐标上，xy轴的单位长度，在实际坐标系中所占据的长度
    return dcm, dpa, ds


def image_orientation(input, orientation_flag=False):  # 判断图片的轴向，输入图片路径或已打开的dicom文件
    try:
        Orientation = input.ImageOrientationPatient
    except:
        try:
            dcm = dicom.read_file(input)
            Orientation = dcm.ImageOrientationPatient
        except:
            print("Please check the image!")
            return None, None
    v1 = [Orientation[0], Orientation[1], Orientation[2]]
    v2 = [Orientation[3], Orientation[4], Orientation[5]]
    n = abs(np.cross(v1, v2))
    if orientation_flag:
        flag = scipy.where(n == max(n))[0][0]  # 2：横截面，1：冠状位，0：矢状位
        return n, flag
    else:
        return n  # 返回图片法向数组,[100]:矢状位,[010]:冠状位,[001]:横截面


def avg_CT_number(image, location, radius):
    """
    to calculate the average pixel value in the image inside a round ROI specified by the location and radius
    location should store the x and y coordinates
    radius the radius of the round ROI
    算由位置和半径指定的圆形ROI内图像的平均像素值
    位置应存储x和y坐标
    半径圆形ROI的半径
    """
    DEBUG = False
    h, w = image.shape
    xc, yc = location
    mask = scipy.mgrid[0:h, 0:w]
    mask = (mask[0] - yc) * (mask[0] - yc) + (mask[1] - xc) * (mask[1] - xc) < radius * radius  # 制造一个掩膜（ROI区域）
    inds = scipy.where(mask)  # ROI区域的位置索引
    CT_values = image[inds]  # 取出ROI区域
    if DEBUG:
        import pylab
        print(scipy.average(CT_values), scipy.std(CT_values), len(CT_values))
        timg = image.copy()
        timg[scipy.where(mask)] = image.min()  # ROI区域归为最小值（最深的颜色）
        pylab.imshow(timg, cmap=pylab.cm.gray)
        pylab.show()
    return scipy.average(CT_values), scipy.std(CT_values)  # 返回均值、方差


avg_CT_value = avg_CT_number

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

integer_types = (np.uint8, np.uint16, np.int8, np.int16)

_supported_types = (np.bool_, np.bool8,
                    np.uint8, np.uint16, np.uint32,
                    np.int8, np.int16, np.int32,
                    np.float32, np.float64)

if np.__version__ >= "1.6.0":
    dtype_range[np.float16] = (-1, 1)
    _supported_types += (np.float16,)  # 值得注意的是，这里的逗号是必需的


def dtype_limits(image, clip_negative=True):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.
    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
        返回图像数据类型的强度限制，即（最小，最大）元组。
    参数：
    image：输入图像
    clip_negative：布尔型变量,如果为真，则剪裁负范围（即返回0表示最小强度）
    即使图像数据类型允许负值。
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels

    Parameters
    ----------
    image : array
        Image you want to smooth.
    function : callable
        A function that does image smoothing.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.

    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
      使用线性函数平滑图像，忽略掩膜像素
参数：
image：希望平滑的图像。
函数：用于进行图像平滑的函数。
掩膜：1代表有效像素，0代表被掩的像素。
提示：
此函数通过将函数应用于遮罩来计算掩膜像素的分数贡献（这将获得由于有效点而产生的像素数据分数）。
然后我们对图像进行掩膜并应用该函数。生成的值将比溢出分数低，因此可以通过除以掩膜上的函数来重新校准，以便仅从有效像素恢复平滑效果。
    """
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask.astype(bool)] = image[mask.astype(bool)]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


def canny(image, sigma=1., low_threshold=None, high_threshold=None, mask=None):
    """Edge filter an image using the Canny algorithm.

    Parameters
    -----------
    image : 2D array
        Greyscale input image to detect edges on; can be of any dtype.
    sigma : float
        Standard deviation of the Gaussian filter.
    low_threshold : float
        Lower bound for hysteresis thresholding (linking edges).
        If None, low_threshold is set to 10% of dtype's max.
    high_threshold : float
        Upper bound for hysteresis thresholding (linking edges).
        If None, high_threshold is set to 20% of dtype's max.
    mask : array, dtype=bool, optional
        Mask to limit the application of Canny to a certain area.

    Returns
    -------
    output : 2D array (image)
        The binary edge map.

    See also
    --------
    skimage.sobel

    Notes
    -----
    The steps of the algorithm are as follows:

    * Smooth the image using a Gaussian with ``sigma`` width.

    * Apply the horizontal and vertical Sobel operators to get the gradients
      within the image. The edge strength is the norm of the gradient.

    * Thin potential edges to 1-pixel wide curves. First, find the normal
      to the edge at each point. This is done by looking at the
      signs and the relative magnitude of the X-Sobel and Y-Sobel
      to sort the points into 4 categories: horizontal, vertical,
      diagonal and antidiagonal. Then look in the normal and reverse
      directions to see if the values in either of those directions are
      greater than the point in question. Use interpolation to get a mix of
      points instead of picking the one that's the closest to the normal.

    * Perform a hysteresis thresholding: first label all points above the
      high threshold as edges. Then recursively label any point above the
      low threshold that is 8-connected to a labeled point as an edge.

    References
    -----------
    Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986

    William Green's Canny tutorial
    http://dasl.mem.drexel.edu/alumni/bGreen/www.pages.drexel.edu/_weg22/can_tut.html

    Examples
    --------
    >>> from skimage import filters
    >>> # Generate noisy image of a square
    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> im += 0.2 * np.random.random(im.shape)
    >>> # First trial with the Canny filter, with the default smoothing
    >>> edges1 = filters.canny(im)
    >>> # Increase the smoothing for better results
    >>> edges2 = filters.canny(im, sigma=3)
    """

    #
    # The steps involved:
    #
    # * Smooth using the Gaussian with sigma above.
    #
    # * Apply the horizontal and vertical Sobel operators to get the gradients
    #   within the image. The edge strength is the sum of the magnitudes
    #   of the gradients in each direction.
    #
    # * Find the normal to the edge at each point using the arctangent of the
    #   ratio of the Y sobel over the X sobel - pragmatically, we can
    #   look at the signs of X and Y and the relative magnitude of X vs Y
    #   to sort the points into 4 categories: horizontal, vertical,
    #   diagonal and antidiagonal.
    #
    # * Look in the normal and reverse directions to see if the values
    #   in either of those directions are greater than the point in question.
    #   Use interpolation to get a mix of points instead of picking the one
    #   that's the closest to the normal.
    #
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #

    if image.ndim != 2:
        raise TypeError("The input 'image' must be a two-dimensional array.")

    if low_threshold is None:
        low_threshold = 0.1 * dtype_limits(image)[1]

    if high_threshold is None:
        high_threshold = 0.2 * dtype_limits(image)[1]

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)
    fsmooth = lambda x: gaussian_filter(x, sigma, mode='constant')
    smoothed = smooth_with_function_and_mask(image, fsmooth, mask)

    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    magnitude = np.hypot(isobel, jsobel)

    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    magnitude = magnitude * eroded_mask

    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape, bool)
    # ----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    #
    # ---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask


def leastsq_circle_fitting(x, y, with_Jacobian=True):  # 圆的拟合
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) 计算每个二维点到中心的距离（xc，yc）"""
        return scipy.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def func(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) 计算数据点与以c=（xc，yc）为中心的平均圆之间的代数距离"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Dfunc(c):
        """ Jacobian of func
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        dfunc_dc = scipy.empty((len(c), x.size))
        Ri = calc_R(xc, yc)
        dfunc_dc[0] = (xc - x) / Ri  # dR/dxc
        dfunc_dc[1] = (yc - y) / Ri  # dR/dyc
        dfunc_dc = dfunc_dc - dfunc_dc.mean(axis=1)[:, scipy.newaxis]
        return dfunc_dc

    # coordinates of the barycenter
    x_m = scipy.mean(x)
    y_m = scipy.mean(y)
    center_estimate = x_m, y_m
    if with_Jacobian:
        center, ier = optimize.leastsq(func, center_estimate, Dfun=Dfunc, col_deriv=True)
    else:
        center, ier = optimize.leastsq(func, center_estimate)

    xc, yc = center
    Ri = calc_R(*center)
    R = Ri.mean()
    residu = scipy.sum((Ri - R) ** 2)
    return xc, yc, R, residu


def find_CT_phantom_outer_edge_threshold(image):  # 寻找CT模体外边缘阈值的函数
    DEBUG = False  # True
    # pixel values on the horizontal and vertical cross lines
    h, w = image.shape
    pvs = scipy.hstack([image[int(h / 2), :], image[:, int(w / 2)].T])  # 将图像中心十字上的像素值提取出来（hstack函数可将多个数组水平铺开）
    # find the most probable pixel value  寻找频率最高的CT值
    hist, le = scipy.histogram(pvs, bins=list(range(int(pvs.min()), int(pvs.max() + 1))))  # hist是频数，le是CT值（从小到大）
    inds = scipy.argsort(hist)  # argsort函数返回排序索引值（从小到大）
    mpv = le[inds[-1]]  # 频数最高的CT值
    # find the pixel value of the air   寻找空气CT值
    pv_air = image[int(h / 2), 5] + image[int(h / 2), -6] + image[5, int(w / 2)] + image[-6, int(w / 2)]
    pv_air /= 4  # 选四个点做一下平均
    # check if the pixel value of the air is the same as the most probable value 检验空气CT值与最多的CT值是否一致
    ind = -1
    while abs(pv_air - mpv) < 400:  # 如果空气CT值和最常见CT值绝对值很接近，则换一个mpv值
        # if yes,
        # change to the next most probable value
        ind -= 1
        mpv = le[inds[ind]]
        print(mpv)
    if DEBUG:
        import pylab
        pylab.hist(pvs, bins=1000, log=True)
        pylab.plot([pv_air] * 2, [hist.min(), hist.max()], 'g-')
        pylab.plot([mpv] * 2, [hist.min(), hist.max()], 'r-')
        pylab.show()
    # the threshold to separate the air and the phantom
    thr = (mpv + pv_air) / 2.  # 阈值为mpv和空气CT值的平均
    return thr  # 返回阈值


def find_CT_phantom_outer_edge(image, pixel_size, return_coors=True, returnEdge=False):
    DEBUG = False
    # assume that the phantom outer shell is composed by materials that is brighter than air     假设模体外壳由比空气更亮的材料组成
    # smoothing the image to ensure noise would not influence the segmentation of the phantom    平滑图像以确保噪声不会影响模体的分割
    ##    smoothed = gaussian_filter(image, 2.0)
    ##    thr = scipy.percentile(smoothed, 45)#55)
    ##    outer = smooth > thr
    thr = find_CT_phantom_outer_edge_threshold(image)  # 求出阈值
    thr = -300  # 这里为什么又指定了一个阈值？
    if DEBUG:
        print("the threshold = %s" % thr)
    if thr == None:
        print("cannot find a phantom in the image!")
        return
    outer = image > thr  # 取出图像中大于阈值的部分

    # get the boundary
    # !!!!!!!!!!!!!! note that the boundary is not in the phantom to avoid some situation
    # !!!!!!!!!!!!!! that only a thin shell was segmented in the previous step,
    # !!!!!!!!!!!!!! in that case, the real "outer" edge may connect to the "inner" edge of the thin shell
    # !!!!!!!!!!!!!! to avoid it, we dilate the segmented image so that the "outer" edge is not possibly
    # !!!!!!!!!!!!!! connected to the "inner" edge
    ##outer_edge = outer- binary_erosion(outer, [[0,1,0],[1,1,1],[0,1,0]])
    outer_edge = scipy.logical_xor(binary_dilation(outer, [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                                   # xor是异或，不相同则为1。将outer膨胀，与原来的的outer异或，得到的结果就是边界
                                   outer)
    lb, nlb = label(outer_edge, scipy.ones((3, 3)))  # lb是连通域，nlb是连通域数目
    ##    # below is in case that a character is at the top edge of the field of view
    ##    for i in range(1,nlb+1):
    ##        count = (lb == i).sum()
    ##        #print count
    ##        if count > 100:
    ##            break

    # find the largest region
    hist, le = scipy.histogram(lb, bins=list(range(1, nlb + 2)))  # hist是频数，le是CT值（从小到大）
    # 用ind=半径
    ##    print hist
    ll = scipy.argsort(hist)[::-1] + 1
    i = scipy.argmax(hist) + 1

    for i in ll:  # 频数最大时的索引加1

        outer_edge = lb == i  # 最外边界层outer_edge
        ys, xs = scipy.where(outer_edge)  # 返回最外层边界的二维索引
        xc, yc, r, rd = leastsq_circle_fitting(xs, ys, with_Jacobian=True)
        print(rd)
        if rd < 500 and (r * pixel_size > 40 and r * pixel_size < 100):
            break
    ##    import cv2
    ##    from cv2 import moments          #计算外边缘质心
    ##    M = cv2.moments(outer_edge+0.0)  #cv2.moments输入必须是浮点型
    ##    xc = int(M["m10"] / M["m00"])
    ##    yc = (max(ys)+min(ys))/2
    ##    r = (max(xs)+min(xs))/2

    # cnt = [(xs[i],ys[i]) for i in range(0,len(xs))]
    # print(cnt)
    # x0,y0,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(outer_edge,(x0,y0),(x0+w,y0+h),(0,255,0),2)
    # cv2.imshow('show',outer_edge)
    if DEBUG:
        import pylab
        pylab.imshow(outer_edge, cmap=pylab.cm.gray)
        pylab.plot(xc, yc, 'r.')
        pylab.show()

    if returnEdge == True:
        return outer_edge
    if return_coors:
        return xc, yc, r, xs, ys  # 返回模体的中心坐标、半径、最外边界点坐标
    else:
        return xc, yc, r


def find_edge_new(image, pixel_size, returnWaterEdge=False):
    DEBUG = False
    thr = -400
    outer = image > thr
    outer_edge = scipy.logical_xor(binary_dilation(outer, [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                                   # xor是异或，不相同则为1。将outer膨胀，与原来的的outer异或，得到的结果就是边界
                                   outer)
    lb, nlb = label(outer_edge, scipy.ones((3, 3)))  # lb是连通域，nlb是连通域数目

    hist, le = scipy.histogram(lb, bins=list(range(1, nlb + 2)))
    i = scipy.argmax(hist) + 1  # 频数最大时的索引加1
    centers = center_of_mass(outer)  # not including background and edges    #找质心 list(range(1, nlb+1))
    ##    print(centers)
    ##    print(i)
    outer_edge = lb == i  # 最外边界层outer_edge
    ys, xs = scipy.where(outer_edge)
    histx, lex = scipy.histogram(xs, bins=list(range(1, len(xs) + 2)))
    histy, ley = scipy.histogram(ys, bins=list(range(1, len(ys) + 2)))
    x1 = scipy.argmax(histx)
    histx_ = histx
    x2 = x1
    while abs(x2 - x1) < 50 / pixel_size:
        histx_[scipy.argmax(histx_)] = min(histx_)
        x2 = scipy.argmax(histx_)

    y1 = scipy.argmax(histy)
    y2 = y1
    histy_ = histy
    while abs(y2 - y1) < 50 / pixel_size:
        histy_[scipy.argmax(histy_)] = min(histy_)
        y2 = scipy.argmax(histy_)

    ##    print(x1,x2,y1,y2)

    if x1 < x2:
        x1, x2 = x2, x1
    if y1 < y2:
        y1, y2 = y2, y1
    x0, y0 = int((x1 + x2) / 2), y1
    if abs(centers[1] - x0) > 15 / pixel_size:
        x0 = int(centers[1])
    print(x0, y0)
    waterx1 = x1
    waterx2 = x2
    watery1 = y1
    watery2 = int(y1 - WATER_DISTANCE / pixel_size)
    ##    print(waterx1,waterx2,watery1,watery2)
    # 整理模体外缘矩形坐标
    lx1 = list(range(x2, x1))
    ly1 = [y1 for m in range(x1 - x2)]
    ly2 = [y2 for m in range(x1 - x2)]
    lx2 = list(range(x2, x1))
    lx3 = [x1 for m in range(y1 - y2)]
    lx4 = [x2 for m in range(y1 - y2)]
    ly3 = list(range(y2, y1))
    ly4 = list(range(y2, y1))
    xs = lx1 + lx2 + lx3 + lx4
    ys = ly1 + ly2 + ly3 + ly4
    # 整理水模体外缘矩形坐标
    lx1 = list(range(waterx2, waterx1))
    ly1 = [watery1 for m in range(waterx1 - waterx2)]
    ly2 = [watery2 for m in range(waterx1 - waterx2)]
    lx2 = list(range(waterx2, waterx1))
    lx3 = [waterx1 for m in range(watery1 - watery2)]
    lx4 = [waterx2 for m in range(watery1 - watery2)]
    ly3 = list(range(watery2, watery1))
    ly4 = list(range(watery2, watery1))
    water_xs = lx1 + lx2 + lx3 + lx4
    water_ys = ly1 + ly2 + ly3 + ly4

    if DEBUG:
        import pylab
        ##        pylab.imshow(array, cmap=pylab.cm.gray)
        ##        pylab.show()
        nplots = 2
        ind = 0
        fig = pylab.figure()
        ind += 1
        ax = pylab.subplot(1, nplots, ind)
        pylab.imshow(image, cmap=pylab.cm.gray)
        ##        pylab.imshow(array, cmap=pylab.cm.gray)
        ys, xs = scipy.where(outer_edge)
        pylab.plot(xs, ys, 'r.')

        ind += 1
        ax = pylab.subplot(1, nplots, ind)
        hst = pylab.hist(image.ravel(), bins=1000, log=True)  # image.ravel(),bins=1000
        pylab.plot([thr] * 2, [0, hst[0].max()], 'r')
        pylab.show()

    if returnWaterEdge:
        return x0, y0, water_xs, water_ys, waterx1, waterx2, watery1, watery2
    else:
        return x0, y0, xs, ys, x1, x2, y1, y2


def find_water_edge(image, pixel_size):
    DEBUG = False
    thr = 50
    outer1 = image > -thr

    outer2 = image < thr

    outer = outer1 & outer2

    lb, nlb = label(outer, scipy.ones((3, 3)))  # lb是连通域，nlb是连通域数目
    hist, le = scipy.histogram(lb, bins=list(range(1, nlb + 2)))
    i = scipy.argmax(hist) + 1
    outer1 = lb == i  #
    y0, x0 = (center_of_mass(outer1))
    y0, x0 = int(y0 + DISTANCE12 / 2), int(x0)
    outer_edge = scipy.logical_xor(binary_dilation(outer1, [[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
                                   # xor是异或，不相同则为1。将outer膨胀，与原来的的outer异或，得到的结果就是边界
                                   outer1)
    lb, nlb = label(outer_edge, scipy.ones((3, 3)))  # lb是连通域，nlb是连通域数目

    hist, le = scipy.histogram(lb, bins=list(range(1, nlb + 2)))
    i = scipy.argmax(hist) + 1
    outer_edge = lb == i  #

    ys, xs = scipy.where(outer_edge)

    histx, lex = scipy.histogram(xs, bins=list(range(1, len(xs) + 2)))
    histy, ley = scipy.histogram(ys, bins=list(range(1, len(ys) + 2)))
    x1 = scipy.argmax(histx)
    histx_ = histx
    x2 = x1
    while abs(x2 - x1) < 50 / pixel_size:
        histx_[scipy.argmax(histx_)] = min(histx_)
        x2 = scipy.argmax(histx_)

    y1 = scipy.argmax(histy)
    y2 = y1
    histy_ = histy
    while abs(y2 - y1) < 50 / pixel_size:
        histy_[scipy.argmax(histy_)] = min(histy_)
        y2 = scipy.argmax(histy_)

    y2 = int(y2 + DISTANCE12 / pixel_size)
    if x1 < x2:
        x1, x2 = x2, x1
    if y1 < y2:
        y1, y2 = y2, y1
    # 整理模体外缘矩形坐标
    lx1 = list(range(x2, x1))
    ly1 = [y1 for m in range(x1 - x2)]
    lx2 = list(range(x2, x1))
    ly2 = [y2 for m in range(x1 - x2)]
    ly3 = list(range(y2, y1))
    lx3 = [x1 for m in range(y1 - y2)]
    ly4 = list(range(y2, y1))
    lx4 = [x2 for m in range(y1 - y2)]

    xs = lx1 + lx2 + lx3 + lx4
    ys = ly1 + ly2 + ly3 + ly4
    if DEBUG:
        import pylab
        pylab.imshow(outer)
        pylab.plot(xs, ys, 'r')
        pylab.show()

    ##    print(x0,y0,x1,y1,x2,y2)
    return (x0, y0, xs, ys, x1, x2, y1, y2)


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;  # 无分号
    C[1, 1] = -1
    E, V = linalg.eig(np.dot(linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    ##    print (up,down1,down2)
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def fit_ellipse(xcoors, ycoors):
    x = np.float64(xcoors)
    y = np.float64(ycoors)
    xmean = x.mean()
    ymean = y.mean()
    x -= xmean
    y -= ymean
    a = fitEllipse(x, y)
    center = ellipse_center(a)  # 椭圆中心坐标
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)  # 椭圆旋转的角度
    axes = ellipse_axis_length(a)  # 椭圆长轴短轴
    x += xmean
    y += ymean
    return center, phi, axes


def tilt_estimate(image_segmented, IsEdge=False, ShowLabel=False):
    """
    return the estimated tile angle of a segmented image
    """
    if not IsEdge:
        img = image_segmented
        edge = img - binary_erosion(img, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        edge = image_segmented
    ys, xs = scipy.where(edge)
    center, phi, axes = fit_ellipse(xs, ys)
    if axes[0] > axes[1]:
        a, b = axes
    else:
        b, a = axes
    if ShowLabel == False:
        return acos(b / a) / pi * 180
    else:
        return acos(b / a) / pi * 180, [xs, ys]


def estimate_tilt_angle(dcm_image, IsFile=False, ShowLabel=False):
    """
        outer = gaussian_filter(image, 4) > 5   #20200614,待解决gaussian_filter(image, 2.0) > 5

        #outer_edge = binary_dilation(outer, [[0,1,0],[1,1,1],[0,1,0]]) - outer
        outer_edge = scipy.logical_xor(binary_dilation(outer, [[0,1,0],[1,1,1],[0,1,0]]),
                                       outer)
        lb, nlb = label(outer_edge, [[1,1,1],[1,1,1],[1,1,1]])

        for i in range(1,3):#(1,3)
            count = (lb == i).sum()
            print (nlb,i,count,count*pixel_size)
            import pylab
            pylab.imshow(lb==i)
            pylab.show()
            if count > 100 and count<2000:
                break

        edge = lb == i
        DEBUG = False
        if DEBUG:
            import pylab
            pylab.imshow(outer)
            pylab.imshow(outer_edge)
            pylab.imshow(lb==1)
            pylab.show()
            pylab.imshow(lb==2)
            pylab.show()
            pylab.imshow(lb==3)
            pylab.show()
    ##    if ShowLabel ==True:
    ##        return [tilt_estimate(edge, IsEdge=True),scipy.where(edge)]
    ##    else:
        return tilt_estimate(edge, IsEdge=True,ShowLabel=ShowLabel)
    pname = u"D:\\Research\\INET\\Du_Guosheng\\data\\Siments_16rows_Hepingli"
    fnames = os.listdir(pname)
##    #dcm = dicom.read_file(os.path.join(pname, '199.Dcm'))
##    dcm = dicom.read_file(os.path.join(pname, '110.Dcm'))
    for i in range(157, len(fnames)):
        fname = os.path.join(pname, fnames[i])
        dcm = dicom.read_file(fname)
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        print "%2d"%i, "%12s"%fnames[i],
        # either way will return the tilt angle in degree
        print estimate_tilt_angle(img),
        print estimate_tilt_angle(fname, IsFile=True)
    """
    if IsFile:
        dcm = dicom.read_file(dcm_image)
        image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        pixel_size = dcm.PixelSpacing[0]
    else:
        image = dcm_image
    ##    outer_edge  = find_CT_phantom_outer_edge(image, pixel_size,return_coors = True,returnEdge = True)
    ##    outer_edge = scipy.logical_xor(binary_dilation(outer_edge, [[0,1,0],[1,1,1],[0,1,0]]),
    ##                                   outer_edge)
    ##    image[scipy.where(outer_edge==True)] = -4000
    mask = canny(image, sigma=5.)  # 5
    ##    print (image)
    lb, nlb = label(mask, scipy.ones((3, 3)))  # [[1,1,1],[1,1,1],[1,1,1]]
    hist, le = scipy.histogram(lb, bins=list(range(1, nlb + 2)))  # hist是频数，le是CT值（从小到大）
    # 用ind=半径
    ##    print hist
    ll = scipy.argsort(hist)[::-1] + 1
    i = scipy.argmax(hist) + 1

    for i in ll:  # 频数最大时的索引加1

        outer_edge = lb == i  # 最外边界层outer_edge
        ys, xs = scipy.where(outer_edge)  # 返回最外层边界的二维索引
        xc, yc, r, rd = leastsq_circle_fitting(xs, ys, with_Jacobian=True)
        print(rd)
        if rd < 500 and (r * pixel_size > 40 and r * pixel_size < 100):
            break

    DEBUG = False
    if DEBUG:
        import pylab
        print(nlb)
        pylab.imshow(mask)
        pylab.show()
        for i in ll:  # 频数最大时的索引加1

            outer_edge = lb == i  # 最外边界层outer_edge
            ys, xs = scipy.where(outer_edge)  # 返回最外层边界的二维索引
            xc, yc, r, rd = leastsq_circle_fitting(xs, ys, with_Jacobian=True)
            print(rd)
            if rd < 500 and (r * pixel_size > 40 and r * pixel_size < 100):
                break
        # for i in range(nlb):
        #     if (lb == i).sum()*pixel_size>200 and (lb == i).sum()*pixel_size<800:
        xs, ys = scipy.where(lb == i)
        xc, yc, r, d = leastsq_circle_fitting(xs, ys, with_Jacobian=True)
        print(d, (lb == i).sum() * pixel_size, i)
        ##                print((lb == i).sum()*pixel_size)
        ##                print(i)
        pylab.imshow(lb == i)
        pylab.show()

    #     for i in range(nlb):
    #         if (lb == i).sum()*pixel_size>200 and (lb == i).sum()*pixel_size<800:
    #             xs,ys = scipy.where(lb==i)
    #             xc, yc, r, d = leastsq_circle_fitting(xs, ys, with_Jacobian=True)
    # ##            print (d)
    #             if d>1000:
    #                 continue
    #             else:
    #                 break

    edge = lb == i
    angle = tilt_estimate(edge, IsEdge=True, ShowLabel=ShowLabel)
    return angle

    ##    outer_edge = scipy.logical_xor(binary_dilation(outer, [[0,1,0],[1,1,1],[0,1,0]]),
    ##                                   outer)

    ##    gg
    ##    center, phi, axes = fit_ellipse(xs, ys)
    ##    if axes[0] > axes[1]:
    ##        a,b = axes
    ##    else:
    ##        b,a = axes
    ##    print(xs.max()-xs.min(),ys.max()-ys.min(),a,b)
    ##    if ShowLabel == False:
    ##        return acos(b/a)/pi*180
    ##    else:
    ##        return acos(b/a)/pi*180,[xs,ys]


def get_id(path):
    f = dicom.read_file(path, stop_before_pixels=True)
    return f.StudyInstanceUID, f.SeriesInstanceUID


def is_dicom_file(path):
    """Fast way to check whether file is DICOM."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(132).decode("ASCII")[-4:] == "DICM"
    except:
        return False


def alphanum_key(s):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]


def dicom_files_in_dir(directory="."):
    """Full paths of all DICOM files in the directory."""
    directory = os.path.expanduser(directory)
    fls = [os.path.join(directory, f) for f in os.listdir(directory)]
    inf = []
    for fname in fls:
        if is_dicom_file(fname):
            ds = dicom.read_file(fname, stop_before_pixels=True)
            try:
                inf.append(
                    {'fname': fname, 'InstanceNumber': int(ds.InstanceNumber), 'SeriesNumber': int(ds.SeriesNumber),
                     'StudyTime': float(ds.StudyTime),
                     'StudyDate': int(ds.StudyDate), 'SeriesDescription': ds.SeriesDescription})
            except:
                inf.append(
                    {'fname': fname, 'InstanceNumber': int(ds.InstanceNumber), 'SeriesNumber': int(ds.SeriesNumber),
                     'StudyTime': float(ds.StudyTime),
                     'StudyDate': int(ds.StudyDate), 'SeriesDescription': 'xx'})

    l = sorted(inf, key=lambda x: (
        x['StudyDate'], x['StudyTime'], x['SeriesNumber'], x['SeriesDescription'],
        x['InstanceNumber']))  # ,x['SeriesNumber']x['SeriesTime'],
    flist = [i['fname'] for i in l]
    return flist  # [f for f in candidates if is_dicom_file(f)]


class DicomData(object):
    def __init__(self, data, **kwargs):
        self._array = data
        self.modality = kwargs.get("modality")

    @classmethod
    def from_files(cls, files):
        data = []
        modality = None

        for file_path in files:
            f = dicom.read_file(file_path)
            print(("Reading %s..." % file_path))

            # Get modality
            if modality:
                if modality != f.Modality:
                    raise RuntimeError("Cannot mix images from different modalities")
            elif f.Modality not in cls.ALLOWED_MODALITIES:
                raise RuntimeError("%s modality not supported" % f.Modality)
            else:
                modality = f.Modality
            data.append(cls._read_pixel_data(f))
        return cls(np.array(data), modality=modality)

    @classmethod
    def _read_pixel_data(cls, f):
        if f.Modality == "CT":
            data = f.RescaleSlope * f.pixel_array + f.RescaleIntercept
            return np.array(data)
        else:
            return np.array(f.pixel_array)

    @property
    def shape(self):
        return self._array.shape

    @property
    def array(self):
        """The underlying numpy array."""
        return self._array

    def get_slice(self, plane, n):
        if plane not in ALLOWED_PLANES:
            raise ValueError("Invalid plane identificator (allowed are 0,1,2)")
        index = [slice(None, None, None) for i in range(3)]
        index[plane] = n
        return self._array[index]

    def get_slice_shape(self, plane):
        # TODO: 
        shape = list(self.shape)
        shape.pop(plane)
        return shape


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # dcm_img = r"D:\\pythonct\py_codenew\new\py_code_latest\py_code_190214_from_Wu\py_code514\old_fromnew\\change3\\0527_体检中心\\Z568"
    # dcm_img = "D:\motituxiang\motituxiang\冠状和矢状-1\\Z01"
    dcm_img = "D:\\医学模体图像\\儿童头模\\儿童头模\\Z42"
    dcm = dicom.read_file(dcm_img, force=True)
    pixel_size = dcm.PixelSpacing[0]
    image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # find_water_edge(image,pixel_size)
    ##    re = find_edge_new(image,pixel_size)
    ##    print(re[2]*dcm.PixelSpacing[0])
    # 加传入dcm.PixelSpacing
    angle = estimate_tilt_angle(dcm_img, IsFile=True)
    print(angle)
