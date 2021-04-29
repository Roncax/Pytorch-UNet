import numpy as np

# return the predicted mask by higher prob value

def combine_predictions(comb_dict, mask_threshold, shape):
    tot = np.zeros(shape)
    for organ1 in comb_dict:
        t = comb_dict[organ1].copy()
        cd = comb_dict.copy()
        cd.pop(organ1)
        for organ2 in cd:
            t[comb_dict[organ1] < comb_dict[organ2]] = 0
        t[t < mask_threshold] = 0
        t[t > mask_threshold] = 1
        tot[t == 1] = organ1
    return tot

def combine_predictions_with_coarse(comb_dict, mask_threshold, shape, coarse):
    tot = np.zeros(shape)
    for organ1 in comb_dict:
        t = comb_dict[organ1].copy()
        cd = comb_dict.copy()
        cd.pop(organ1)
        for organ2 in cd:
            t[comb_dict[organ1] < comb_dict[organ2]] = 0

        t[t < mask_threshold] = 0
        t[t > mask_threshold] = 1
        tot[t == 1] = organ1

    return tot


labels = {"0": "Bg",
          "1": "RightLung",
          "2": "LeftLung",
          "3": "Heart",
          "4": "Trachea",
          "5": "Esophagus",
          "6": "SpinalCord"
          }
mask_threshold = 0.5
comb_dict = {}
shape = (20,20)

for l in labels:

    t = np.round(np.random.rand(20,20), 2)
    comb_dict[l]=t
    print(f"label  {l}")
    print(t)

tot = combine_predictions_with_coarse(comb_dict=comb_dict, mask_threshold=mask_threshold, shape=shape)
print(f"Total: \n{tot}")