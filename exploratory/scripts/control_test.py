import socket
import time

host = 'localhost'
port = 42425
s = socket.socket()
s.connect((host,port))


import dronekit_sitl
sitl = dronekit_sitl.start_default()
connection_string = sitl.connection_string()

#mavproxy.exe --master tcp:127.0.0.1:5770 --sitl 127.0.0.1:5501 --out 127.0.0.1:14550 --out 127.0.0.1:14551

# Import DroneKit-Python
from dronekit import connect, VehicleMode

# Connect to the Vehicle.
print("Connecting to vehicle on: %s" % (connection_string,))
vehicle = connect("127.0.0.1:14550", wait_ready=True)

print(vehicle.battery)
print(vehicle.is_armable)


vehicle.close()
sitl.stop()

for i in range(100):
    time.sleep(0.25)
    s.send("NEXT_STATE".encode("utf-8"))
    data = s.recv(1024)
    print(data)
