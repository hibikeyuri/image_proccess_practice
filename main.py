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
def showIMG(img, windowName):
    cv.namedWindow(windowName, cv.WINDOW_NORMAL)
    cv.resizeWindow(windowName, 800, 600)
    cv.imshow(windowName, img)
    cv.waitKey(300)
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
imgray_o = cv.cvtColor(imo, cv.COLOR_BGR2GRAY)
imgray_b = cv.cvtColor(imb, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(imgray_o, (11, 11), 0)
blurred_b = cv.GaussianBlur(imgray_b, (25, 25), 0)

kernel = np.ones((21, 21), dtype="uint8")
closing_b = cv.morphologyEx(imgray_b, cv.MORPH_OPEN, kernel)

edged_o = cv.Canny(blurred, 70, 210)
edged_b = cv.Canny(blurred_b, 70, 210)
#showIMG(edged, "edged sample")
contours_o, hierarchy_o = cv.findContours(edged_o.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
contours_b, hierarchy_b = cv.findContours(closing_b.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
"""

"""
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

if __name__ == '__main__':


    images = []
    image_file = None
    img_path = "img/168677272_480702383078432_6365521997202246368_n.jpg"

    """    
    for thres in range(180):
        temp_img = binarize_img(img_path, thres)
        images.append(temp_img)
        print("Proccessing image for threshold : {}".format(thres))
    
    make_gif_PIL(images, "pillow_imagedraw")
    """
    

    threshold = 70
    tmp_result = binarize_img(img_path, threshold)
    tmp_result.save("result.jpg")

    if os.path.isdir("./gif"):
        print("dir is exist")
    else:
        print("creating directory...")
        os.mkdir("./gif")

    imo = cv.imread(img_path)
    imb = cv.imread("result.jpg")

    contours_o, hierarchy_o = preprocessing_IMG(img_path)
    contours_cb, hierarchy_cb = preprocessing_IMG("result.jpg", method="morph", morph="OPEN", kernel=(21, 21))

    sample_o = imo.copy()
    sample_cb = imb.copy()

    contours = contours_cb
    contour_cb_size = [cv.contourArea(contour) for contour in contours]
    contours_cb = sorted(contours, key=cv.contourArea ,reverse=False)[0:24]
    sorted_cnts = sorted(contours_cb, key=lambda cnt: cv.boundingRect(cnt)[1])
    cnts = sorted_cnts

    for i in range(4):
        part_cnts = sorted(cnts[6*i: 6*i+6], key=lambda cnt: cv.boundingRect(cnt)[0])
        for index, cnt in enumerate(part_cnts):
            cv.drawContours(sample_cb, part_cnts, index, (0, 0, 255), 2)
            cv.imwrite("./gif/{}.jpg".format(i*6 + (index + 1)), sample_cb)
            print("Contour #{}".format(i*6 + (index + 1)))
            #showIMG(sample_b, "GOGOGO")

    make_gif("./gif", "contour_drawing")
    delete_dir("./gif")
    
