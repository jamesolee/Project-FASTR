import time
import serial

port = serial.Serial(port='/dev/ttyAMC0', baudrate=115200)

# port.write(bytes(b'usbsd\n'))
# port.write(bytes(b'setmapping2 YUYV 640 480 30.0 JeVois SerialTest\n'))
port.write(bytes(b'setpar serout USB\n'))
port.write(bytes(b'setmapping2 YUYV 640 480 15.0 JeVois SerialTest\n'))
time.sleep(5)
port.write(bytes(b'streamon\n'))
print("yuh")

while True:
    string = port.readline()
    print(string)