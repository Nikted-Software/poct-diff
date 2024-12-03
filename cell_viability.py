import cv2
from threshold_sauvola import sau
import math
import matplotlib.pyplot as plt

image_path = "62388edd-5f75-4063-8d2b-fddf59ffdc14.jpg"
image = cv2.imread(image_path)
grad = sau(image, 2001, 1.5)
cv2.imshow("window_name", grad)
cv2.waitKey(0)
contours, _ = cv2.findContours(grad, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cont = []
for contour in contours:
    if cv2.contourArea(contour) >= 10000:
        cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)
        cont.append(contour)
cv2.imwrite("viability1.jpg", image)

image = cv2.imread(image_path)
grad = sau(image, 15, 0.2)
cv2.imshow("window_name", grad)
cv2.waitKey(0)
cv2.imwrite("viability2.jpg", grad)
circulars = []
size = []
contours, _ = cv2.findContours(grad, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        for contour1 in cont:
            dist = cv2.pointPolygonTest(contour1, (cx, cy), True)
            if dist > 0:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circular = (4 * math.pi * cv2.contourArea(contour)) / (perimeter**2)
                    circulars.append(circular)
                else:
                    circular = 0
                size.append(cv2.contourArea(contour))
                if (
                    cv2.contourArea(contour) > 60
                    and cv2.contourArea(contour) < 5000
                    and circular > 0
                ):
                    print(circular)
                    cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)


fig1, ax1 = plt.subplots()
ax1.hist(circulars, density=True, bins=10, range=[0, 1])
ax1.set_xlabel("size value")
ax1.set_ylabel("population")
plt.show()
# plt.savefig("size.png")
plt.close(fig1)

fig2, ax2 = plt.subplots()
ax2.hist(size, density=True, bins=50, range=[0, 500])
ax2.set_xlabel("size value")
ax2.set_ylabel("population")
plt.show()
# plt.savefig("size.png")
plt.close(fig2)
cv2.imwrite("viability.jpg", image)
