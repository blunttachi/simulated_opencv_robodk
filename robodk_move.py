from robodk import robolink  # RoboDK API
from robodk import robodialogs  # Dialogs
from robodk.robomath import *

items = [{"item" : "Zdenka", "pic" : "test3\item3.jpg", "height" : 20}, 
        {"item" : "Nescafe", "pic" : "test3\item1.jpg", "height" : 65},
        {"item" : "Metar", "pic" : "test3\item4.jpg", "height" : 40},
        {"item" : "Crta", "pic" : "test3\item2.jpg", "height" : 47}]

def runmain():
    RDK = robolink.Robolink()
    RDK.setSimulationSpeed(1)
    robot = RDK.Item("UR5")
    home = RDK.Item("Home", 6)
    basket = RDK.Item("Basket",6)
    center = RDK.Item("Center",6)
    robot.MoveJ(home)

    target_dist = []

    for item in items:
        target = RDK.Item(item["item"], 6)
        pose = Pose_2_TxyzRxyz(target.Pose())
        distance = sqrt(pose[0]**2 + pose[1]**2)
        target_dist.append({"item" : item["item"], "distance" : distance, "height" : item["height"]})

    sorted_targets = sorted(target_dist, key=lambda d: d['distance']) 

    for item in sorted_targets:
        target = RDK.Item(item["item"], 6)
        print(item)
        robot.MoveJ(target)
        robot.MoveL(target.Pose()*transl(0,0,200-item["height"]+30))
        robot.setDO(1, 1)
        robot.MoveL(target)
        robot.MoveJ(center)
        robot.MoveJ(basket)
        robot.setDO(1,0)
        robot.MoveJ(center)
        
    robot.MoveJ(home)

if __name__ == '__main__':
    runmain()