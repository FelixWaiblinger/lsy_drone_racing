def reset(c):
    c += 1
    if (c % 10) == 0:
        print("seed")
        c = 0
    return c

def episode(i):
    print("episode", i)

c = -1
for i in range(100):
    c = reset(c)
    episode(i)

OFFSETS = \
[[[ 0.11434896,-0.13407594, 0.525     ],
[-0.06436689,-0.00177486, 1.        ],
[-0.11679738,-0.13017636, 0.525     ],
[0.14153467,0.01389439,1.        ],],

[[0.05391394,0.06246483,0.525     ],
[0.00371492,0.1012571 ,1.        ],
[-0.09888389 ,0.05166066 ,0.525     ],
[0.02117119,0.06940641,1.        ],],

[[-0.03803965 ,0.08774895 ,0.525     ],
[-0.13312612 ,0.08065671 ,1.        ],
[ 1.18357249e-01,-4.15545566e-05 ,5.25000000e-01],
[-0.03542189 ,0.11917431 ,1.        ],],

[[-0.13483227, 0.14642245 ,0.525     ],
[ 0.12905499,-0.04843381 ,1.        ],
[-0.10902309,-0.14930831 ,0.525     ],
[ 0.03036089,-0.12230242 ,1.        ],],

[[-0.11548854,-0.0740628 , 0.525     ],
[-0.02203566,-0.09269406, 1.        ],
[-0.05363571, 0.06790475, 0.525     ],
[0.13186599,0.0251885 ,1.        ],],

[[-0.01270655 ,0.12583684, 0.525     ],
[ 0.02423145,-0.13049215, 1.        ],
[-0.11158907, 0.09692389, 0.525     ],
[ 0.07626643,-0.14587904, 1.        ],],

[[ 0.07965028,-0.11110329, 0.525     ],
[ 0.04953184,-0.03940619, 1.        ],
[0.01685447,0.07950789,0.525     ],
[-0.13427407,-0.07437305, 1.        ],],

[[ 0.10750898,-0.07752962 ,0.525     ],
[-0.02136714 ,0.06825233 ,1.        ],
[0.01854879,0.09203467,0.525     ],
[-0.04783239 ,0.13114597 ,1.        ],],

[[-0.06236039,-0.06338777 ,0.525     ],
[0.1351116 ,0.10807759,1.        ],
[-0.0777331 ,-0.07535889 ,0.525     ],
[-0.02232176 ,0.12297617 ,1.        ],],

[[ 0.02001983,-0.09586169 ,0.525     ],
[-0.00250008 ,0.1089074  ,1.        ],
[-0.0946377 , 0.05302584 ,0.525     ],
[0.02247221,0.02013733,1.        ],],]

OBST = \
[[[ 0.07761377,-0.1422427  ,0.525     ],
[-0.06430704, 0.10308778 ,0.525     ],
[-0.04153058,-0.05502561 ,0.525     ],
[0.12562772,0.07002324,0.525     ],],

[[-0.09193052,-0.10352564 ,0.525     ],
[-0.12245035,-0.11863344 ,0.525     ],
[0.02430012,0.1037509 ,0.525     ],
[ 0.14412109,-0.0493231  ,0.525     ],],

[[-0.00645187 ,0.1461663  ,0.525     ],
[-0.08875955 ,0.06728223 ,0.525     ],
[0.03582431,0.05992261,0.525     ],
[ 0.12593631,-0.125385   ,0.525     ],],

[[-0.12865972,-0.0783465  ,0.525     ],
[0.00970189,0.06919654,0.525     ],
[0.0882966 ,0.11136996,0.525     ],
[-0.05548951 ,0.04774376 ,0.525     ],],

[[-0.0626068  ,0.05425257 ,0.525     ],
[-0.04181516,-0.13240968 ,0.525     ],
[-0.0086212 ,0.1240647 ,0.525    ],
[-0.1230042  ,0.05609347 ,0.525     ],],

[[-0.01156004,-0.0581216  ,0.525     ],
[-0.10058794 ,0.13606317 ,0.525     ],
[-0.12190188 ,0.11050593 ,0.525     ],
[ 0.04674863,-0.09475442 ,0.525     ],],

[[-0.11672545 ,0.10289274 ,0.525     ],
[-0.11931768,-0.11246388 ,0.525     ],
[0.11959747,0.0721842 ,0.525     ],
[-0.04755191 ,0.01349858 ,0.525     ],],

[[ 0.12092295,-0.06030219 ,0.525     ],
[ 0.04919136,-0.02260313 ,0.525     ],
[ 0.0335752 ,-0.01777377 ,0.525     ],
[-0.02601975,-0.03573427 ,0.525     ],],

[[0.10691531,0.09597654,0.525     ],
[0.01809035,0.04879842,0.525     ],
[-0.04037895,-0.00929128 ,0.525     ],
[ 0.06614783,-0.11808482 ,0.525     ],],

[[-0.07077377 ,0.06911782 ,0.525     ],
[ 0.12557805,-0.11439094 ,0.525     ],
[ 0.1365461 ,-0.02543756 ,0.525     ],
[-0.12505196 ,0.11099108 ,0.525     ],],]

POSE = \
[[[0.    ,    0.     ,   0.10308778],
[ 0.   ,      0.   ,     -0.11038707],
[ 0.   ,      0.   ,     -0.04153058],
[ 0.   ,      0.   ,     -0.05502561],]
,
[[ 0.   ,      0.   ,     -0.03359982],
[ 0.   ,      0.   ,     -0.12379421],
[ 0.   ,      0.   ,     -0.14553579],
[0.    ,    0.     ,   0.02239338],]
,
[[ 0.   ,      0.   ,     -0.03827043],
[0.    ,    0.     ,   0.11295191],
[0.    ,    0.     ,   0.12959635],
[0.    ,    0.     ,   0.02123786],]
,
[[ 0.   ,      0.   ,     -0.03657939],
[0.    ,    0.     ,   0.12403608],
[ 0.   ,      0.   ,     -0.08129648],
[0.    ,    0.     ,   0.02927316],]
,
[[0.    ,    0.     ,   0.11768415],
[ 0.   ,      0.   ,     -0.14015445],
[ 0.   ,      0.   ,     -0.08638568],
[ 0.   ,      0.   ,     -0.01902636],]
,
[[ 0.   ,      0.   ,     -0.14816427],
[ 0.   ,      0.   ,     -0.12194189],
[0.    ,   0.      , 0.0632291],
[ 0.   ,      0.   ,     -0.03243054],]
,
[[ 0.   ,      0.   ,     -0.13712257],
[ 0.   ,      0.   ,     -0.09302151],
[0.    ,    0.     ,   0.00898524],
[0.    ,    0.     ,   0.12506211],]
,
[[ 0.   ,      0.   ,     -0.09985938],
[ 0.   ,      0.   ,     -0.09517125],
[0.    ,    0.     ,   0.04774218],
[ 0.   ,      0.   ,     -0.05912026],]
,
[[0.    ,    0.     ,   0.07985696],
[ 0.   ,      0.   ,     -0.04832714],
[0.    ,    0.     ,   0.11363663],
[0.    ,    0.     ,   0.11672702],]
,
[[ 0.   ,      0.   ,     -0.08493381],
[ 0.   ,      0.   ,     -0.10682937],
[0.    ,    0.     ,   0.04822316],
[ 0.   ,      0.   ,     -0.14104966]]]
