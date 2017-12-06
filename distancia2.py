import cv2
import numpy as np



DISTANCIA_CONOCIDA = 41
ANCHO_CONOCIDO = 15.7
DISTANCIA_FOCAL = 513.26

#Captura 5 fotos para luego buscar el area mas grande y calcular la distancia

#encuentra el area de la cual se medira la distancia

def find_objeto(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 200)
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        return cv2.minAreaRect(c)
    else:
        return []


def distancia_a_camara(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


def seguimiento_objeto():
    captura = cv2.VideoCapture(0)
    while(captura.isOpened()):
        _, frame = captura.read()
        objeto = find_objeto(frame)
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