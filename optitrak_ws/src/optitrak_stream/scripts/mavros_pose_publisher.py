#!usr/bin/env python3
import rospy
from mavros_msgs.msg import Mavlink

if __name__ == '__main__':
    rospy.init_node("mavros_pose_publisher")
    rospy.loginfo("optitrak data is streaming")

    pub = rospy.Publisher("/mavlink/to", Mavlink, queue_size = 10)

    #refresh rate at which information is published
    rate = rospy.Rate(15)

    while not rospy.is_shutdown():
        #TODO stream optitrak data to RPi

        msg = Mavlink()
        #TODO fill in required mavlink msg data

        pub.publish(msg)

        rate.sleep()