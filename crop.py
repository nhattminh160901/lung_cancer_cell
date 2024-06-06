import cv2

# load the original input image and display it on our screen
image = cv2.imread("croplinesDetected.jpg")
h, w, _ = image.shape
h1 = int(h/4)
w1 = int(w/4)
for i in range(4):
    for j in range(4):
        x = int(w1*j)
        y = int(h1*i)
        crop = image[y:y+h1, x:x+w1]
    
        dim = (640, 640)
        # perform the actual resizing of the image
        resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
        path = "images/"+str(i+1)+str(j+1)+".jpg"
        cv2.imwrite(path, resized)
