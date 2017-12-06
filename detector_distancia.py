import cv2
import numpy as np


DISTANCIA_CONOCIDA = 41
ANCHO_CONOCIDO = 15.7
DISTANCIA_FOCAL = 509.23
# verdesBajos = (29, 86, 6)
# verdesAltos = (64, 255, 255)

rojosBajos1 = np.array([0, 65, 75], dtype=np.uint8)
rojosAltos1 = np.array([12, 255, 255], dtype=np.uint8)
rojosBajos2 = np.array([240, 65, 75], dtype=np.uint8)
rojosAltos2 = np.array([256, 255, 255], dtype=np.uint8)


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
        return cv2.minAreaRect(c)
    else:
        return []

def distancia_a_camara(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def seguimiento_objeto():
    captura = cv2.VideoCapture(1)
    while(captura.isOpened()):
        _, frame = captura.read()
        objeto = buscador_objeto(frame)
        if objeto:
            box = np.int0(cv2.boxPoints(objeto))
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            distancia = distancia_a_camara(ANCHO_CONOCIDO, DISTANCIA_FOCAL, objeto[1][0])
            cv2.putText(frame, " % .2f m" % (distancia / 100), (10, 400)
            , cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

        """ cv2.putText(frame, " % .2f m" % (distancia / 100), (10, 400)
        , cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
"""
        cv2.imshow("ventana", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    captura.release()
    cv2.destroyAllWindows


seguimiento_objeto()
