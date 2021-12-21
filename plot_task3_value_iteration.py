import numpy as np
import matplotlib.pyplot as plt

'''
policy iteration starts here
'''
policy = np.array([[[[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]],


       [[[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]],


       [[[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]],


       [[[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]],


       [[[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]]])

value_function = np.array([[[[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]]],


       [[[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.97013031, -0.95779682, -0.95223075, -0.94517571,
          -0.92992133, -0.91900027],
         [-0.87354039, -0.8622661 , -0.84870417, -0.84479574,
          -0.84490695, -0.8494538 ],
         [-0.84621145, -0.8595921 , -0.8713064 , -0.87205733,
          -0.88303393, -0.8882151 ],
         [-0.93136457, -0.95111678, -0.95431812, -0.96402036,
          -0.96637056, -0.97616819],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.94429182, -0.92705194, -0.91551072, -0.88735559,
          -0.86464472, -0.83104585],
         [-0.74650555, -0.68947595, -0.65954821, -0.64022909,
          -0.62899405, -0.62444923],
         [-0.65833702, -0.70914672, -0.73444572, -0.75065433,
          -0.78302754, -0.79423992],
         [-0.88073896, -0.91551757, -0.92473074, -0.93263284,
          -0.94848804, -0.96486996],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.94864678, -0.92213113, -0.90582567, -0.87931496,
          -0.85482804, -0.81377664],
         [-0.71444424, -0.63221083, -0.58604401, -0.5627962 ,
          -0.56396858, -0.55259859],
         [-0.59351355, -0.66433581, -0.69499422, -0.72631753,
          -0.75235862, -0.77313205],
         [-0.86952005, -0.90742942, -0.91947563, -0.93349065,
          -0.93927173, -0.96194496],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]]],


       [[[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.94816514, -0.92775868, -0.91417766, -0.90066312,
          -0.87562698, -0.85273103],
         [-0.76843531, -0.72536364, -0.69899225, -0.68858303,
          -0.68559   , -0.67491981],
         [-0.6998945 , -0.73457262, -0.75479112, -0.77251188,
          -0.787724  , -0.80345403],
         [-0.88348894, -0.91467828, -0.92351404, -0.93714997,
          -0.94888975, -0.96088871],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.94453445, -0.92017693, -0.91054348, -0.88045576,
          -0.84093187, -0.80986394],
         [-0.70959156, -0.6376939 , -0.60662516, -0.58349341,
          -0.56470071, -0.56622692],
         [-0.60083734, -0.66545835, -0.70710308, -0.7326939 ,
          -0.75581801, -0.77427308],
         [-0.86342774, -0.90265282, -0.92276625, -0.93310743,
          -0.94648905, -0.96462977],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.95350442, -0.93143322, -0.91782904, -0.89475069,
          -0.86554803, -0.8452307 ],
         [-0.76433504, -0.71985979, -0.69237298, -0.67140872,
          -0.66310524, -0.65795683],
         [-0.67946484, -0.72392543, -0.74658271, -0.76466323,
          -0.78211956, -0.79799389],
         [-0.88143391, -0.91123919, -0.92620599, -0.93214783,
          -0.9471894 , -0.96424773],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]]],


       [[[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.9456893 , -0.91837627, -0.89995478, -0.88618636,
          -0.85230389, -0.82162856],
         [-0.72447298, -0.6547742 , -0.62774663, -0.60698365,
          -0.59149272, -0.58639423],
         [-0.62313607, -0.68319934, -0.71053738, -0.73546956,
          -0.75854556, -0.77543049],
         [-0.86526389, -0.90863874, -0.92258216, -0.92973857,
          -0.94518235, -0.96058903],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.94336417, -0.91798892, -0.89816499, -0.8721465 ,
          -0.84836291, -0.80558719],
         [-0.69793634, -0.60588138, -0.56510399, -0.54566645,
          -0.53311461, -0.52743849],
         [-0.57710599, -0.65069375, -0.69384461, -0.72501451,
          -0.75087964, -0.76975213],
         [-0.86469483, -0.90651529, -0.92408559, -0.93420466,
          -0.94464534, -0.96127265],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [-0.96880961, -0.9547881 , -0.95083976, -0.93721489,
          -0.93025668, -0.91280306],
         [-0.87135893, -0.85930741, -0.85020928, -0.84596854,
          -0.83845863, -0.84063575],
         [-0.84162102, -0.85319233, -0.86317178, -0.86664765,
          -0.87682165, -0.88477013],
         [-0.93262372, -0.94715865, -0.95256547, -0.96157751,
          -0.96411817, -0.97665879],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]]],


       [[[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,
           0.        ,  0.        ]]]])

# plot here
'''
Plots of value function. Since there are 4 state variables, generate plots of value function with
respect to theta and x for selected theta' and x' . At least three plots are needed.
'''
X1 = [-3, -1.6, 0, 1.6, 3] * 6 # POSITION
X=[]
for i in range(5):
  for j in range(6):
    X.append(X1[i])
Y = [-18, -9, -3, 3, 9, 18] * 5 # angle
Z = value_function[:, 2, :, 3]
Z=Z.reshape((1,-1))[0]
fig = plt.figure()
plt.title("Given theta' btw -10 and 10 degree/s and x' greater than 5")
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', label='value')
ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('value')

fig = plt.figure()
plt.title("Given theta' btw -30 and -10 degree/s and x' btw -0.5 and 5")
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', label='x')
ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('value')


fig = plt.figure()
plt.title("Given theta' btw -50 and -30 degree/s and x' greater than 5")
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', label='x')
ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('value')

'''
Plots of optimal policy, in the same way as in the plots of value function.
'''
Z = policy[:, 2, :, 3]
Z=Z.reshape((1,-1))[0]
fig = plt.figure()
plt.title("Given theta' btw -10 and 10 degree/s and x' greater than 5")
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', label='x')
ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('policy')

fig = plt.figure()
ax = fig.add_subplot(211)
plt.title("Given theta' btw -30 and -10 degree/s and x' btw -0.5 and 5")
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', label='x')
ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('policy')

fig = plt.figure()
plt.title("Given theta' btw -50 and -30 degree/s and x' greater than 5")
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c='red', label='x')
ax.set_xlabel('x')
ax.set_ylabel('theta')
ax.set_zlabel('policy')
plt.show()