#!/usr/bin/env python

'''
test mavlink messages
'''

from __future__ import print_function

from builtins import object

from pymavlink.dialects.v10 import ardupilotmega as mavlink1

from pymavlink import mavutil

import time

class fifo(object):
    def __init__(self):
        self.buf = []
    def write(self, data):
        self.buf += data
        return len(data)
    def read(self):
        return self.buf.pop(0)

def test_protocol(mavlink, signing=False):
    # we will use a fifo as an encode/decode buffer
    f = fifo()

    print("Creating MAVLink message...")
    # create a mavlink instance, which will do IO on file object 'f'
    mav = mavlink.MAVLink(f)

    if signing:
        mav.signing.secret_key = bytearray(chr(42)*32, 'utf-8' )
        mav.signing.link_id = 0
        mav.signing.timestamp = 0
        mav.signing.sign_outgoing = True

    # set the WP_RADIUS parameter on the MAV at the end of the link
    mav.param_set_send(7, 1, b"WP_RADIUS", 101, mavlink.MAV_PARAM_TYPE_REAL32)

    # alternatively, produce a MAVLink_param_set object 
    # this can be sent via your own transport if you like
    m = mav.param_set_encode(7, 1, b"WP_RADIUS", 101, mavlink.MAV_PARAM_TYPE_REAL32)

    m.pack(mav)

    # get the encoded message as a buffer
    b = m.get_msgbuf()

    bi=[]
    for c in b:
        bi.append(int(c))
    print("Buffer containing the encoded message:")
    print(bi)

    print("Decoding message...")
    # decode an incoming message
    m2 = mav.decode(b)

    # show what fields it has
    print("Got a message with id %u and fields %s" % (m2.get_msgId(), m2.get_fieldnames()))

    # print out the fields
    print(m2)

print("Testing mavlink1\n")
test_protocol(mavlink1)

from argparse import ArgumentParser
parser = ArgumentParser(description=__doc__)

parser.add_argument("--baudrate", type=int,
                  help="master port baud rate", default=115200)
parser.add_argument("--device", required=True, help="serial device")
parser.add_argument("--source-system", dest='SOURCE_SYSTEM', type=int,
                  default=255, help='MAVLink source system for this GCS')
args = parser.parse_args()

def wait_heartbeat(m):
    '''wait for a heartbeat so we know the target system IDs'''
    print("Waiting for APM heartbeat")
    msg = m.recv_match(type='HEARTBEAT', blocking=True)
    print("Heartbeat from APM (system %u component %u)" % (m.target_system, m.target_component))

# create a mavlink serial instance
drone = mavutil.mavlink_connection(args.device, baud=args.baudrate, source_system=args.SOURCE_SYSTEM)

# wait for the heartbeat msg to find the system ID
wait_heartbeat(drone)

print("Sending all message types")
# mavtest.generate_outputs(master.mav)


# ARM THROTTLE
drone.mav.command_int_send(drone.target_system,drone.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,0,1,0,0,0,0,0,0)

# TAKEOFF 0.5
drone.mav.command_int_send(drone.target_system,drone.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,0,0,0.5)
msg = drone.recv_match(type='COMMAND_ACK', blocking=True)
print(msg)

time.sleep(10)

# LAND
drone.mav.command_int_send(drone.target_system,drone.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mavutil.mavlink.MAV_CMD_NAV_LAND,0,0,0,0,0,0,0,0,0)
msg = drone.recv_match(type='COMMAND_ACK', blocking=True)
print(msg)


time.sleep(10)


# DISARM
drone.mav.command_int_send(drone.target_system,drone.target_component,mavutil.mavlink.MAV_FRAME_LOCAL_NED,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,0,0,0,0,0,0,0,0)
msg = drone.recv_match(type='COMMAND_ACK', blocking=True)
print(msg)
