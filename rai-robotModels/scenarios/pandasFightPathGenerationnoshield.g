world: {}

### table

origin (world): { Q: [0, 0, .6], shape: marker, size: [.03] }
table (origin): { Q: [0, 0, -.05], shape: ssBox, size: [2.3, 2, .1, .02], color: [.3, .3, .3], contact, logical:{ } }

## two pandas
Prefix: "l_"
Include: <../panda/panda.g>
Prefix: "r_"
Include: <../panda/panda.g>
Prefix: False

## position them on the table
Edit l_panda_base (origin): { Q: "t(-0.5 0 0) d(0 0 0 1)", motors, joint: rigid }
Edit r_panda_base (origin): { Q: "t(0.5 0 0) d(180 0 0 1)", motors, joint: rigid }

## make gripper dofs inactive (unselected)
Edit l_panda_finger_joint1: { joint_active: False}
Edit r_panda_finger_joint1: { joint_active: False}

### camera

camera(world): {
 Q: "t(-0.01 -.2 2.) d(-150 1 0 0)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.5, 100]
}

sword_0 (l_gripper):{
    shape: ssBox,
    size: [0.1, .03, 0.03, 0.001],
    joint: rigid,
    Q: "t(0 0 0) d(90 0 0 1)",
    mass: 0.1,
    contact:1,
    color: [0.05, 0.05, 0.05,1]
    }

sword_1 (sword_0):{
    shape: ssBox,
    size: [0.4, .02, 0.02, 0],
    Q: "t(0.25 0 0)",
    mass: 0.1,
    contact: 1,
    color: [0.05, 0.05, 0.05,1],
    'restitution': 0.4
}

sword_2 (sword_0):{
    shape: ssBox,
    size: [0.04, .08, 0.05, 0],
    Q: "t(0.05 0 0.0)",
    mass: 0.1,
    contact: 1,
    color: [0.05, 0.05, 0.05,1]}

sword_3 (sword_0):{
    shape: ssBox,
    size: [0.04, .08, 0.05, 0],
    Q: "t(-0.05 0 0.0)",
    mass: 0.1,
    contact: 1,
    color: [0.05, 0.05, 0.05,1]}

