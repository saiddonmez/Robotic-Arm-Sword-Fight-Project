## create standard base frame before including

panda_base: { multibody: true }
Include: <panda_cleanfull.g>
Edit panda_link0(panda_base): {}

## simpler collision models

#Delete panda_link0_0:
#Delete panda_link1_0:
#Delete panda_link2_0:
#Delete panda_link3_0:
#Delete panda_link4_0:
#Delete panda_link5_0:
#Delete panda_link6_0:
#Delete panda_link7_0:
#Delete panda_hand_0:
#Delete panda_leftfinger_0:
#Delete panda_rightfinger_0:

panda_coll0(panda_link0): { shape: capsule, color: [1.,1.,1.,.1], size: [.1, .11], Q: "t(-.04 .0 .03) d(90 0 1 0)", contact: -2 }
#panda_coll0b(panda_link0): { shape: capsule, color: [1.,1.,1.,.1], size: [.2, .06], Q: "t(-.2 -.12 .0) d(90 0 1 0)", contact: -2 }

panda_coll1(panda_joint1): { shape: capsule, color: [1.,1.,1.,.1], size: [.2, .08], Q: "t(0 0 -.15)", contact: -2 }
panda_coll3(panda_joint3): { shape: capsule, color: [1.,1.,1.,.1], size: [.2, .08], Q: "t(0 0 -.15)", contact: -2 }
panda_coll5(panda_joint5): { shape: capsule, color: [1.,1.,1.,.1], size: [.22, .09], Q: "t(0 .02 -.2)", contact: -2 }

panda_coll2(panda_joint2): { shape: capsule, color: [1.,1.,1.,.1], size: [.12, .12], Q: "t(0 0 .0)", contact: -2 }
panda_coll4(panda_joint4): { shape: capsule, color: [1.,1.,1.,.1], size: [.12, .08], Q: "t(0 0 .0)", contact: -2 }
panda_coll6(panda_joint6): { shape: capsule, color: [1.,1.,1.,.1], size: [.1, .07], Q: "t(0 .0 -.04)", contact: -2 }
panda_coll7(panda_joint7): { shape: capsule, color: [1.,1.,1.,.1], size: [.1, .07], Q: "t(0 .0 .01)", contact: -2 }

## zero position

Edit panda_joint1: { q: 0.0 }
Edit panda_joint2: { q: -.5 }
Edit panda_joint3: { q: 0. }
Edit panda_joint4: { q: -2. }
Edit panda_joint5: { q: -0. }
Edit panda_joint6: { q: 2., limits: [.5, 3.] }
Edit panda_joint7: { q: -.5 }
Edit panda_finger_joint1: { q: .05 }
        
## kill rigid hand joints

## deactivate the gripper joint by default! control and simulation treats it differently
Edit panda_finger_joint1: { joint_active: False }

mug(panda_link8): { shape: ssBox, size: [0.2, 0.03, 0.2, 0.005],  
    joint: rigid, 
    Q: "t(0 0 0.02) d(90 1 0 0) d(180 0 0 1)",
    contact: -5, 
    color: [0.80, 0.80, 0.80]
    }