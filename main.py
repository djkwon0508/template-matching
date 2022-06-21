import cv2
import glob

row = 10
col = 6

image_path = r''
template_path = r''
images = glob.glob(image_path)
templates = glob.glob(template_path)


# NCC Using OpenCV
def NCC_matching_template(image, template, imgDST):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = maxLoc
    h, w = template.shape

    imgDST = cv2.cvtColor(imgDST, cv2.COLOR_BGR2RGB)
    imgDST = cv2.rectangle(imgDST, (x, y), (x + w, y + h), (255, 0, 0), 4)
    # Use under code if your image doesn't change BGR -> RGB
    # imgDST = imgDST[...,::-1]

    return result, imgDST


# MAIN
def main():
    for tplNum in templates:
        storage = []
        cnt = 1

        print('\n')
        print(tplNum)
        templ = cv2.imread(tplNum, cv2.IMREAD_GRAYSCALE)
        # Resize if you want
        templ = cv2.resize(templ, dsize=(400, 350))
        for imgNum in images:
            print(imgNum)
            img = cv2.imread(imgNum, cv2.IMREAD_GRAYSCALE)
            dst = cv2.imread(imgNum, cv2.IMREAD_COLOR)

            # NCC Function
            corrImg, matchImg = NCC_matching_template(img, templ, dst)
            storage.append(matchImg)
            cnt += 1

        # Show image
        cv2.imshow("Correlation Image", corrImg)
        for i in range(0, cnt - 1):
            name = "Matching Image " + str(i + 1)
            cv2.imshow(name, storage[i])
        cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
