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


shi(r_gripper): { shape: ssCylinder, size: [0.02, 0.17,0.01],    
    mass: 0.1, 
    joint: rigid, 
    Q: "t(0 0 -0.05)",    
    contact: 1, 
    color: [0.36, 0.20, 0.20,1],
    'restitution':0.5
    }

shi-h1(shi): { shape: ssBox, size: [0.01, 0.091, 0.05, 0.001],    
    mass: 0.1,    
    Q: "t(0 0 0.034) d(90 1 0 0)",  
    contact: 1,
    color: [0.10, 0.40, 0.40,1],
    }

shi-h2(shi): { shape: ssBox, size: [0.05, 0.091, 0.01, 0.001],     
    mass: 0.1,   
    Q: "t(0 0.0155 0.034) d(90 1 0 0)",  
    contact: 1, 
    color: [0.10, 0.40, 0.40,1],
    }

shi-h3(shi): { shape: ssBox, size: [0.05, 0.091, 0.01, 0.001],     
    mass: 0.1,   
    Q: "t(0 -0.0155 0.034) d(90 1 0 0)",  
    contact: 1, 
    color: [0.10, 0.40, 0.40,1],
    }