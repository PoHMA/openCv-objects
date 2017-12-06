import cv2
import numpy as np

rojosBajos1 = np.array([0, 65, 75], dtype=np.uint8)
rojosAltos1 = np.array([12, 255, 255], dtype=np.uint8)
rojosBajos2 = np.array([240, 65, 75], dtype=np.uint8)
rojosAltos2 = np.array([256, 255, 255], dtype=np.uint8)
DISTANCIA_CONOCIDA = 41
ANCHO_CONOCIDO = 15.7

captura = cv2.VideoCapture(1)
while(True):
    _, frame = captura.read()
    cv2.imshow("ventana", frame)
    if cv2.waitKey(1) & 0xFF == ord('f'):
        cv2.imwrite("images/img_dist.jpg", frame)
        print("se tomo la foto")
        break
captura.release()
cv2.destroyAllWindows

def buscador_objeto(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(gray, 50, 200)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv = cv2.GaussianBlur(hsv, (11,11), 0)
    mascara_rojo1 = cv2.inRange(hsv, rojosBajos1, rojosAltos1)
    mascara_rojo2 = cv2.inRange(hsv, rojosBajos2, rojosAltos2)
    # mask = cv2.inRange(hsv, verdesBajos, verdesAltos)
    mask = cv2.add(mascara_rojo1, mascara_rojo2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        print(c)
        return cv2.minAreaRect(c)
    else:
        return []

image = cv2.imread("images/img_dist.jpg")
marker = buscador_objeto(image)

focalLength = (marker[1][0] * DISTANCIA_CONOCIDA) / ANCHO_CONOCIDO
print(focalLength)
