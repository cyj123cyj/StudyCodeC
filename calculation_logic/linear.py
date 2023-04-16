import os
import pydicom as dicom
import scipy
import math
import pydicom
# from _canny_edge import canny
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import label
from utils.util import canny
from utils.util import avg_CT_number
from utils.util import find_CT_phantom_outer_edge

# from misc import find_CT_phantom_outer_edge

# this depends on the design of the phantom
### we determined this number empirically
##DISK_ENCLOSE_RODS = 3190. #mm^2
### changed by Yahui 08/28/17
### due to the fact that the design of the phantom was changed
##DISK_ENCLOSE_RODS = 800. #mm^2
##ROD_AREA = 140. #mm^2
##ROD_AREA = 20. #mm^2
##DISTANCE_ROD2CENTER = 50. #mm^2
##
# below are the accurate numbers
ROD_RADIUS = 7.5  # mm  the designed size is 7.5 mm          棒的实际半径
PERCENTAGE_ROD = 0.64  # the percentage of the ROD area in CT value calculation, 80% as instructed in WS 519-2016
ROD_AREA = ROD_RADIUS * ROD_RADIUS * scipy.pi  # mm^2            棒的实际面积
DISTANCE_ROD2CENTER = 50.  # mm
# there is another design, where the distance between the rod center and the phantom center is 55 mm
# therefore, it is safe to extend the distance by 5 mm.
# and the extra 2 mm is to ensure rods are all included for sure
# the same tolerate limit is used to filter out the rod for MTF
TOLERATE_LIMIT = 6.  # mm
RADIUS_ENCLOSE_RODS = (DISTANCE_ROD2CENTER + ROD_RADIUS + 5 + TOLERATE_LIMIT)  # mm       能够覆盖所有棒的半径


def find_rod_locations(image, pixelspacing, returnR=False):  # 第二个参数是util.read_dicom_file中获得的spacing值
    """
    determine the locations of the 8 rods in the CT value linearity phantom
    assuming the geometric parameters are known, one may identify rod edges first inside a circle
    not all rod edges can be identified because the contrast may not be large enough
    since we know the rods are distributed evenly on a circle, we may fill in the missing rods
    by checking the angles of rods relative to the center of the phantom
    the locations of the rods are returned
    确定CT值线性体模中8根棒的位置
    假设几何参数已知，可以首先在圆内识别杆边
    由于对比度可能不够大，所以无法识别所有的杆边缘
    因为我们知道这些杆均匀地分布在一个圆上，所以我们可以通过检查杆相对于模型中心的角度把丢失的杆补上
    返回值：杆的位置
    """
    # 8 rods distributed evenly on a circle   八个棒在一个圆上均匀分布
    DEBUG = False
    # pixel size in mm2
    pixelarea = pixelspacing * pixelspacing  # 一个像素的面积
    # calculate the detection area (the disk area containing rods) and rod area in pixels
    detection_dist = RADIUS_ENCLOSE_RODS / pixelspacing  # 检测距离(像素）
    rod_size = ROD_AREA / pixelarea * .5  # .9 is a factor to ensure the rods will be detected even with defects

    rod2center_distance = DISTANCE_ROD2CENTER / pixelspacing  # 棒与圆心的像素距离
    if DEBUG:
        print("disk radius that includes the rods: %f (pixels), %f (mm)" % (
        detection_dist, detection_dist * pixelspacing))
        print("minimum area of a rod: %f (pixels), %f (mm^2)" % (rod_size, ROD_AREA))
        rp, rm = [math.sqrt(e / scipy.pi) for e in [rod_size, rod_size * pixelarea]]
        print("    effective radius: %f (pixels), %f (mm)" % (rp, rm))
        print("distance from the phantom center to the rods: %f (pixels), %f (mm)" % (
        rod2center_distance, rod2center_distance * pixelspacing))
    # get the cener of the phantom
    xc, yc, r = find_CT_phantom_outer_edge(image, pixelspacing, return_coors=False)  # 找出有效部分，将其拟合成圆，得到圆心坐标、半径
    # this has to be fixed, a temporary solution
    ##    from water import Water_Phantom
    ##    phan = Water_Phantom(dcm_image_file)
    ##    outer = phan.get_phantom_outer_boundary()

    ##    outer = image > 100
    ##    ys, xs = scipy.where(outer)
    ##    yc, xc = ys.mean(), xs.mean()
    ##    r = math.sqrt(outer.sum() / scipy.pi)
    ##    print "center of the phantom:" ,xc ,yc
    ##    print "radius of the outer boundary:", r

    # find a restricted area that contains the rods
    # a mask that hightlight the restricted area
    h, w = image.shape
    mask = scipy.mgrid[0:h, 0:w]  # mgrid函数用来返回多维结构
    mask = scipy.hypot((mask[0] - yc), (mask[1] - xc)) < detection_dist  # 取出在检测范围内的部分

    # canny edge filtering inside the restricted area
    edge = canny(image, sigma=2, low_threshold=50, high_threshold=100)  # 用canny算子求出边界
    edge *= mask  # 取出边界中的有效部分
    if DEBUG:
        import pylab
        pylab.imshow(edge)
        pylab.show()
    # identify a few regions (rod candidates) through the closed edges
    lb, nlb = label(edge == 0)
    centers = center_of_mass(edge == 0, lb, index=list(range(2, nlb + 1)))  # not including background and edges    #找质心
    sizes, le = scipy.histogram(lb.ravel(),
                                bins=list(range(2, nlb + 2)))  # count no background and edges      ravel函数可以将多维数组转化为一维
    if DEBUG:
        import pylab
        print(centers)
        print(sizes)
        print("rod size threshold: %s" % rod_size)
        print("rod center to phantom center threshold: %s" % rod2center_distance)
        pylab.imshow(lb >= 2)
        pylab.show()
        pylab.imshow(lb)
        pylab.show()
    # rank the rods by size descendingly
    inds = scipy.argsort(sizes)[::-1]
    szs = sizes[inds]  # region sizes                               区域的大小
    les = le[inds]  # and label IDs are sorted                 区域的位置
    # filter the regions by size (if too small then not a rod)
    inds = scipy.where(szs > rod_size)[0]  # 选择比棒面积大的区域
    ##    print len(inds)
    ##    print inds
    rod_cand_ids = les[inds]

    rod_rs, rod_angles, rod_ids = [], [], []
    for i in rod_cand_ids:
        # print i
        c_ind = i - 2  # the indices of center of mass and region sizes are not the same as the region label  质量中心和区域大小的索引与区域标签不同
        r = scipy.hypot(centers[c_ind][0] - yc, centers[c_ind][1] - xc)  # 选取的区域与圆心的距离
        ##        # was used to check the accuracy of the rod positions
        ##        print "%8.2f mm or %8.2f pixels"%(r*pixelspacing, r)
        # check whether it is on the circle
        # error tolerant limit is set to 2 mm
        if DEBUG:
            import pylab
            print("distance to the center:", r)
            print("expected distance:", rod2center_distance)
            print("tolerance:", TOLERATE_LIMIT / pixelspacing)
        if scipy.absolute(r - rod2center_distance) > TOLERATE_LIMIT / pixelspacing:  # 倘若r与棒心像素距离的绝对值在容许范围内则继续
            continue
        if DEBUG:
            print("the distance from rod center to phantom center is", r)
            print("the difference from the expected value is", scipy.absolute(r - rod2center_distance))
            print("the effective rod radius is ", math.sqrt(sizes[c_ind] / scipy.pi))
        t = math.atan2(centers[c_ind][0] - yc, centers[c_ind][1] - xc)  ## * 180 / scipy.pi                      #算区域的角度
        rod_rs.append(r)
        rod_angles.append(t)
        rod_ids.append(i)  # 记录计算出的半径、角度
        if DEBUG:
            print(centers[c_ind], sizes[c_ind], end=' ')
            print(r, t)
            pylab.imshow(lb == i)
            pylab.show()

    if DEBUG:
        print("rod centers to the phantom center:")
        print(rod_rs)
        print("angular position of rod centers:")
        print(rod_angles)
        print("rod label IDs:")
        print(rod_ids)
    # rank the rods by the angular position
    inds = scipy.argsort(rod_angles)  # 按照角度排序
    rod_rs = scipy.array(rod_rs)[inds]
    rod_angles = scipy.array(rod_angles)[inds]
    rod_ids = scipy.array(rod_ids)[inds]

    # determine which rods are missed
    rod_rs = list(rod_rs)
    rod_angles = list(rod_angles)  # 将半径和角度列表化
    if DEBUG:
        print("sorted rod positions:")
        print(rod_rs)
    while rod_angles[0] > -7 / 8. * scipy.pi:  # 角度大于-3/4π时
        rod_angles.insert(0, rod_angles[0] - 45 / 180. * scipy.pi)  # 在列表第一个处加
        rod_rs.insert(0, rod_rs[0])
    while rod_angles[-1] < 7 / 8. * scipy.pi:
        rod_angles.append(rod_angles[-1] + 45 / 180. * scipy.pi)
        rod_rs.append(rod_rs[-1])  # 此段两个while总之要保证第一个是-3/4π
    # print(rod_angles)
    diff_angle = [e[0] - e[1] for e in zip(rod_angles[1:], rod_angles[:-1])]  # 计算每两个角度之间的差值（基本都是π/4）
    newloc = scipy.where(scipy.array(diff_angle) > 3 / 8. * scipy.pi)  # 返回差值大于π/3的索引（说明中间有个棒对比度太低没找出来)
    if DEBUG:
        print("angle difference:")
        print(diff_angle)
        print("location indices of missing rods:")
    for i in newloc[0][::-1]:  # 此处将对比度不够高的棒找出来
        if diff_angle[i] < 5 / 8. * scipy.pi:  # 加1个棒的情况
            ##        if i == 0:
            ##            break
            r = (rod_rs[i] + rod_rs[i + 1]) / 2
            a = (rod_angles[i] + rod_angles[i + 1]) / 2
            rod_rs.insert(i + 1, r)
            rod_angles.insert(i + 1, a)
        if diff_angle[i] > 7 / 8. * scipy.pi and diff_angle[i] < 9 / 8. * scipy.pi:  # 加3个棒的情况
            r = (rod_rs[i] + rod_rs[i + 1]) / 2
            a1 = rod_angles[i] / 4 * 3 + rod_angles[i + 1] / 4
            a2 = (rod_angles[i] + rod_angles[i + 1]) / 2
            a3 = rod_angles[i] / 4 + rod_angles[i + 1] / 4 * 3
            rod_rs.insert(i + 1, r)
            rod_angles.insert(i + 1, a1)
            rod_rs.insert(i + 2, r)
            rod_angles.insert(i + 2, a2)
            rod_rs.insert(i + 3, r)
            rod_angles.insert(i + 3, a3)
        if diff_angle[i] > 5 / 8. * scipy.pi and diff_angle[i] < 7 / 8. * scipy.pi:  # 加2个棒的情况
            r = (rod_rs[i] + rod_rs[i + 1]) / 2
            a1 = rod_angles[i] / 3 * 2 + rod_angles[i + 1] / 3
            a2 = rod_angles[i] / 3 + rod_angles[i + 1] / 3 * 2
            rod_rs.insert(i + 1, r)
            rod_angles.insert(i + 1, a1)
            rod_rs.insert(i + 2, r)
            rod_angles.insert(i + 2, a2)
        if diff_angle[i] > 9 / 8. * scipy.pi:  # 加4个棒的情况
            r = (rod_rs[i] + rod_rs[i + 1]) / 2
            a1 = rod_angles[i] / 5 * 4 + rod_angles[i + 1] / 5
            a2 = rod_angles[i] / 5 * 3 + rod_angles[i + 1] / 5 * 2
            a3 = rod_angles[i] / 5 * 2 + rod_angles[i + 1] / 5 * 3
            a4 = rod_angles[i] / 5 + rod_angles[i + 1] / 5 * 4
            rod_rs.insert(i + 1, r)
            rod_angles.insert(i + 1, a1)
            rod_rs.insert(i + 2, r)
            rod_angles.insert(i + 2, a2)
            rod_rs.insert(i + 3, r)
            rod_angles.insert(i + 3, a3)
            rod_rs.insert(i + 4, r)
            rod_angles.insert(i + 4, a4)
    if DEBUG:
        print("All rods (in polar coordinates):")
        print(rod_rs)
        print(rod_angles)
        print(len(rod_angles))
        print(len(rod_angles) > 8)

    # in case there are more than 8 locations, there must be repeated rod centers
    # we just need to detect them and then remove them
    if len(rod_angles) > 8:
        diffs = scipy.array(rod_angles[1:]) - rod_angles[:-1]  # 此段代码用于检测出超过8个区域时，删除角度查小于π/6的两个，代替以两个的均值
        inds = scipy.where(diffs < scipy.pi / 6)[0]
        for ind in inds[::-1]:
            new_angle = (rod_angles[ind] + rod_angles[ind + 1]) / 2.
            new_rod_r = (rod_rs[ind] + rod_rs[ind + 1]) / 2.
            rod_angles.pop(ind + 1)  # pop函数用于移除
            rod_rs.pop(ind + 1)
            rod_angles[ind] = new_angle
            rod_rs[ind] = new_rod_r

    # modified 5/17/2019
    # in case there are more than 8 rod locations
    # the above code can only remove repeated rods whose angles are too close
    # but cannot exclude rods whose angle appear twice at both pi and -pi
    if rod_angles[0] < -7 / 8. * scipy.pi and rod_angles[-1] > 7 / 8. * scipy.pi:
        rod_angles = rod_angles[:8]  # 这个写法的含义是直接删去第8个之后的
        rod_rs = rod_rs[:8]

    if DEBUG:
        print(rod_angles)
        print(rod_rs)

    rod_locations = (xc + rod_rs * scipy.cos(rod_angles),
                     yc + rod_rs * scipy.sin(rod_angles))
    x, y = rod_locations  # 计算出8个棒的圆心坐标
    DEBUG = False
    if DEBUG:
        import pylab
        # notice that, by default, the origin of the image is at the "lower" left
        pylab.imshow(image, cmap=pylab.cm.gray, origin='upper')
        pylab.plot(x, y, 'rx')
        pylab.show()
    if returnR == True:
        return rod_rs
    return x, y  # 返回8个棒的圆心坐标


# alias of the function
# avg_CT_number = avg_CT_number

class Linearity_Phantom:  # 模体线性度类
    def __init__(self, dicom_image_file):  # 例化时输入文件名
        """
        The dicom image is assumed to image a water phantom
        """
        self.dicom_file = dicom_image_file
        self.LabelPos = None
        try:
            ds = dicom.read_file(dicom_image_file)  # 读取dicom文件
        except pydicom.errors.InvalidDicomError:
            # print "Please specify a valid DICOM image file!"
            return
        self.dataset = ds
        # print "dicom slope and intercept:", ds.RescaleSlope, ds.RescaleIntercept
        self.image_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept  # 得到DICOM图像
        self.pixel_spacing = ds.PixelSpacing[0]  # 得到spacing值

    ##        pylab.imshow(self.image_array)
    ##        pylab.show()

    def get_phantom_center(self):  # 调用找到拟合圆圆心的方法
        xc, yc, r = find_CT_phantom_outer_edge(self.image_array, self.pixel_spacing, return_coors=False)
        # print "center of the phantom: ", xc, yc
        return xc, yc, r

    def get_rod_locations(self):  # 调用找到棒位置的方法
        x, y = find_rod_locations(self.image_array, self.dataset.PixelSpacing[0])
        # x, y = find_rod_locations(self.dicom_file, self.dataset.PixelSpacing[0])
        self.LabelPos = [x, y]
        print(self.LabelPos)
        return x, y

    def get_material_CT_values(self):
        xc, yc, phantom_r = self.get_phantom_center()
        rod_locx, rod_locy = self.get_rod_locations()
        img = self.image_array #没用到变量建议删掉

        CT_values = []
        rod_radius = ROD_RADIUS / self.dataset.PixelSpacing[0]  # in pixels   棒的像素半径
        effective_radius = rod_radius * math.sqrt(PERCENTAGE_ROD)  # ROI is smaller than the actual ROD size
        for i in range(len(rod_locx)):  # should be 8
            CT, std = avg_CT_number(self.image_array, (rod_locx[i], rod_locy[i]), effective_radius)  # 计算棒区的均值方差
            CT_values.append(CT)
            # print "CT value = %8.2f; std = %8.2f"%(CT, std)

        return CT_values


class DICOM:
    def __init__(self, pn, seq, im, studyID):
        self.pn = pn
        self.seq = seq
        self.im = im
        self.studyID = studyID

    def Read(self, displayImage=False):
        dcm_dir = dicom.read_dicomdir(os.path.join(self.pn, "DICOMDIR"))
        seqs = dcm_dir.DirectoryRecordSequence
        for seqInd in range(len(seqs)):
            if seqs[seqInd].DirectoryRecordType == 'STUDY' and int(seqs[seqInd].StudyID) == self.studyID:
                for i in range(seqInd, len(seqs)):
                    if seqs[i].DirectoryRecordType == 'SERIES' and int(seqs[i].SeriesNumber) == self.seq:
                        # print 'Found the Series'
                        for j in range(i, len(seqs)):
                            # print 'Examing#%d '%j, seqs[j].DirectoryRecordType
                            if seqs[j].DirectoryRecordType == 'IMAGE' and int(seqs[j].InstanceNumber) == self.im:
                                # print 'Found the Image, Seq #%d' %j
                                fname = os.path.join(self.pn, os.path.join(*seqs[j].ReferencedFileID))
                                return fname


if __name__ == '__main__':
    # first, find a suitable CT image that must include the linearity test phantom
    # pn = 'D:\\Research\\INET\\Du_Guosheng\\data\\validation_dataset\\hepingli_hospital\\4-19\\4-19\\BSoft'
    # pn = '..\\BSoft'
    # seq = 1
    # im = 1
    # studyID = 1
    # displayImage = True
    #
    # dicomIns = DICOM(pn, seq, im, studyID)
    # fname = dicomIns.Read(True)

    #    pn = "D:\Research\INET\Du_Guosheng\data\\11-4.8\\11-4.8"
    #    pn = "D:\\Research\\INET\\Du_Guosheng\\data\\19220674-GE-RGRMS-E\\19220674-GE-RGRMS-E\\2-5.0mm"
    pn = "F:\\CT_phantom_pictures"
    files = os.listdir(pn)  # 打开整个文件夹
    fnames = [os.path.join(pn, e) for e in files]  # 遍历全部文件
    for fname in fnames[311:312]:  # 注意这里必须用综合模体，不然会报错
        print(fname)

        # instantiate first
        test = Linearity_Phantom(fname)  # 例化模体线性度类
        # then call a function to get all the CT values of the 8 rods
        x = test.get_material_CT_values()  # x include the eight CT value                         #计算出8个棒的CT值
        # print out the CT values sorted for the lowest to the highest
        print(sorted(x))  # 输出排序好的CT值
