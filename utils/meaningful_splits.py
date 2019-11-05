# from models.st_gcn_ae import get_exp_classes

# Kinetics splits:
splits = dict()
splits['lifters'] = [19, 59, 88, 134, 318]           # 0
splits['pullups'] = [183, 255, 260, 305, 330]        # 1
splits['skiing']  = [280, 306, 310, 360]             # 2
splits['dancing'] = [18, 43, 75, 283, 348]           # 3
splits['batting'] = [142, 143, 161, 245, 246]        # 4
splits['throwing'] = [148, 166, 208, 298, 356, 358]  # 5
splits['music'] = [217, 223, 225, 230, 232, 234]     # 6
splits['riding'] = [268, 269, 270, 273]              # 7
splits['swimming'] = [339, 340, 341]                 # 8
splits['jumping'] = [151, 172, 182, 367]             # 9
splits['cycling'] = [267, 271, 275, 375]


splits['rand1'] = [6, 77, 254, 336]   # Completely random
splits['rand2'] = [68, 170, 192, 289, 375]
splits['rand3'] = [141, 105, 274, 307, 253]
splits['rand4'] = [229, 303, 394, 148, 19]
splits['rand5'] = [60, 163, 375, 387]
splits['rand6'] = [14, 141, 176, 183]
splits['rand7'] = [99, 235,  42,  28]
splits['rand8'] = [246, 220, 330]
splits['rand9'] = [143, 378, 170]
splits['rand10'] = [172, 254, 179]

# Split of useful classes without any classes from the splits for
# pretraining the Many-vs-few (245-vs-5 model)
# splits_concat = [i for sl in list(splits.values()) for i in sl]
# _, pretrain_classes = get_exp_classes(splits_concat)
# splits['pretrain'] = pretrain_classes

# NTU Splits:
ntu_splits = dict()
# Classes from https://github.com/shahroudy/NTURGB-D
ntu_splits['arms'] = [31, 38, 39, 40]
ntu_splits['brushing'] = [1, 3, 4]
ntu_splits['dressing'] = [14, 15, 16, 17]
ntu_splits['dropping'] = [5, 6, 8, 9]
ntu_splits['glasses'] = [18, 19, 20, 21]
ntu_splits['handshaking'] = [55, 56, 57, 58]
ntu_splits['office'] = [28, 29, 30, 33]
ntu_splits['pushing'] = [50, 51, 52, 53]
ntu_splits['touching'] = [44, 45, 46, 47]
ntu_splits['waving'] = [10, 23, 31, 38]


ntu_splits['nrand1'] = [14, 24, 38, 58]   # Completely random
ntu_splits['nrand2'] = [0,  7, 18, 55]
ntu_splits['nrand3'] = [11, 26, 32, 53]
ntu_splits['nrand4'] = [2, 12, 32, 43]
ntu_splits['nrand5'] = [29, 33, 34, 52]
ntu_splits['nrand6'] = [2, 19, 21, 24]
ntu_splits['nrand7'] = [5, 13, 18, 39]
ntu_splits['nrand8'] = [43, 51, 54, 54]
ntu_splits['nrand9'] = [18, 34, 43, 50]
ntu_splits['nrand10'] = [7, 10, 49, 56]
ntu_splits['nrand11'] = [3, 31, 35, 38]
ntu_splits['nrand12'] = [3, 13, 14, 42]

