import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
import os 


"""PIL part"""
def make_gif_PIL(imgs, gifname, duration=30):
    imgs[0].save(gifname + ".gif", format='GIF', save_all=True, append_images=images[1:], duration=duration, loop=0)

        
def binarize_img(img_path, threshold):
    """img converted to numpy.array object"""
    image_file = Image.open(img_path)
    image_file = image_file.convert('L')
    image_file = np.array(image_file)
    image_binarized = binarize_array(image_file, threshold)
    image_final = Image.fromarray(image_binarized)

    #image_draw = ImageDraw.Draw(image_final)
    #image_draw.text((50, 50), "threshold : {}".format(threshold))
    return image_final


def binarize_array(img_array, threshold):
    """B&W with different threshold set by users"""
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            if img_array[i][j] > threshold:
                img_array[i][j] = 255
            else:
                img_array[i][j] = 0
    
    return img_array


"""OpenCV part"""
def showIMG(img, windowName, duration=300):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowName, 800, 600)
    cv.imshow(windowName, img)
    cv.waitKey(duration)
    cv.destroyAllWindows()


def preprocessing_IMG(img_path, method="blur", morph="OPEN", kernel=(11, 11)):
    im = cv.imread(img_path)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    if method == "blur":
        blurred = cv.GaussianBlur(imgray, kernel, 0)
        edged = cv.Canny(blurred, 70, 210)
        contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
        return contours, hierarchy
    elif method == "morph" and morph=="OPEN":
        kernel = np.ones(kernel, dtype="uint8")
        opening = cv.morphologyEx(imgray, cv.MORPH_OPEN, kernel)
        contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
        return contours, hierarchy
    else:
        print("Other situiation")
    

"""
#using contour methods to mask user wanted pictures

for i, cnt in enumerate(cnts):
    (x, y, w, h) = cv.boundingRect(cnt)
    print("Sample #{}".format(i+1))
    sample = sample_o[y:y+h, x:x+w]
    mask = np.zeros(imgray_b.shape[:2], dtype="uint8")
    ((centerX, centerY), radius) = cv.minEnclosingCircle(cnt)
    cv.circle(mask, (int(centerX), int(centerY)), int(radius), (255, 255, 255), -1)
    mask = mask[y:y+h, x:x+w]
    showIMG(cv.bitwise_and(sample, sample, mask=mask), "sample + mask")
"""

def make_dir(dirname):
    if os.path.isdir(dirname):
        print("dir is exist")
    else:
        print("creating directory...")
        os.mkdir(dirname)

def delete_dir(dirname):
    try:
        for file in os.listdir(dirname):
            os.remove(dirname + "/" + file)
    except OSError as e:
        print(f"Error:{ e.strerror}")
    
    if(os.path.exists(dirname)):
        os.rmdir(dirname)
        print("directory is removed.")
    else:
        print("directory is not exit.")
    


def make_gif(path, gifname, duration=100):
    """many images converted to GIF"""
    imgs = []
    fileslist = os.listdir(path)
    filelist_ordered = sorted(fileslist, key=lambda x: int(x[:-4]), reverse=False)
    try:
        for file in filelist_ordered:
            filepath = path + '/' + file
            if os.path.isfile(filepath):
                print("Append file : {}".format(filepath))
                image_file = Image.open(filepath)
                imgs.append(image_file)
            else:
                print("{} is not a file".format(filepath))
    except:
        print("PermissionError")
    
    imgs[0].save(gifname + ".gif", format='GIF', save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def houghcircle_processing_IMG(img_path, mask=False):
    img = cv.imread(img_path, 0)
    img_copy = img.copy()
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=250, param2=30, minRadius=70, maxRadius=100)
    circles = np.uint16(np.around(circles))

    """
    make_dir("./gif")

    for index, i in enumerate(circles[0, :]):
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        print("drawing #{} circle".format(index + 1))
        cv.imwrite("./gif/{}.jpg".format(index + 1), cimg)
        #cv.imwrite("houghmethod.jpg", cimg)

    #showIMG(cimg, "detected circles")
    #cv.imwrite("houghmethod.jpg", cimg)
    make_gif("./gif", "houghcirclegif")
    delete_dir("./gif")
    """

    make_dir("./gif")

    if mask is True:
        for index, i in enumerate(circles[0, :]):
            i = i.astype(int)
            crop = img_copy[i[1] - i[2]:i[1] + i[2], i[0] - i[2]:i[0]+i[2]] #[y-r:y+r, x-r: x+r]
            mask = np.zeros(crop.shape, dtype="uint8") # remember to set dtype = uint8!
            mask = cv.circle(mask, (i[2], i[2]), i[2], (255, 255, 255), -1)
            #showIMG(mask, "mask")
            res = cv.bitwise_and(crop, crop, mask=mask)
            cv.imwrite("./gif/{}.jpg".format(index + 1), res)
            #showIMG(res, "masked")
            print("masking #{} circle".format(index + 1))
    
    #make_gif("./gif", "maskedgif")
    #delete_dir("./gif")


def preprocessing_Slicing_IMG(img_path, kernel=(1, 1), threshold=70):
    target_img = binarize_img(img_path, threshold=threshold)
    target_img.save(img_path)
    target_img = cv.imread(img_path)
    target_img_copy = target_img.copy()
    contours, hierarchy = preprocessing_IMG(img_path, "blur", "open", kernel=kernel)
    return contours, hierarchy


def counting_numbers(contours, eraseout=True):
    if eraseout == True:
        contours_area = [cv.contourArea(contour) for contour in contours]
        contours_area = contours_area[:-1]
        return len(contours_area)
    


def counting_area(contours, eraseout=True):
    if eraseout == True:
        contours_area = [cv.contourArea(contour) for contour in contours]
        area_total = sum(contours_area)
        return area_total


def counting_area_ratio(areas, relative="bacteria_area", sample_area=None):
    if relative is not "sample_area":
        ratio_area = [area / max(areas) for area in areas]
        return ratio_area
    else:
        ratio_area = [(area / sample_area) for area in areas]
        return ratio_area


def main():
    pass


"""    
for thres in range(180):
    temp_img = binarize_img(img_path, thres)
    images.append(temp_img)
    print("Proccessing image for threshold : {}".format(thres))

make_gif_PIL(images, "pillow_imagedraw")
"""


if __name__ == '__main__':


    #images = []
    #image_file = None
    #img_path = "img/168458087_302883851253433_4112756444715094781_n.jpg"
    #img_path = "img/168677272_480702383078432_6365521997202246368_n.jpg"
    #img_path = "result.jpg"
    
    #houghcircle_processing_IMG(img_path, True)
    

    #hangle with each sample : counting the # of dots and areas

    areas = []
    dir_path = "./gif"
    filelist = os.listdir(dir_path)
    for file in filelist:
        filepath = path + '/' + file
        if os.path.isfile(filepath):
            print("Append file : {}".format(filepath))
            contours, hierarchy = preprocessing_Slicing_IMG(filepath)
            areas.ap
            counts = counting_numbers(contours)
            area = counting_area(contours)


        else:
            print("{} is not a file".format(filepath))


"""
    sample_path = "./gif/17.jpg"
    tmp_img = binarize_img(sample_path, 90) #讀取路徑
    tmp_img.save("./gif/17_tmp.jpg")
    tmp_img = cv.imread("./gif/17_tmp.jpg") #讀取路徑
    #showIMG(tmp_img, "tmp", 1000)
    sample = cv.imread("./gif/17_tmp.jpg")
    sample_copy = sample.copy()
    contours, hierarchy = preprocessing_IMG(sample_path, "blur", "open", (1, 1))
    cv.drawContours(sample_copy, contours, -1, (0, 255, 0), 2)
    contours_area = [cv.contourArea(contour) for contour in contours]
    total = sum(contours_area)
    nested_contours = [x for x in contours_area if x > 0]
    print(sorted(nested_contours))
    print("AREA: {}, LEN: {}".format(total, len(nested_contours)))
    #showIMG(sample_copy, "tmp", 10000)
"""

#思路整理
#1.讀取切片後的圖片資料夾
#2.進行二值化
#3.獲取contours之後對contours進行排序
#4.如果
