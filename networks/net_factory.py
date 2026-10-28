from networks.network_BraTS import MI2Net_BraTS
from networks.network_MyoPS import MI2Net_MyoPS

def net_factorys(net_type="unet", in_chns=1, class_num=4, mode = "train"):

    if net_type == "MI2Net_BraTS" and mode == "train":
        net = MI2Net_BraTS(num_cls=class_num).cuda()
    elif net_type == "MI2Net_BraTS" and mode == "test":
        net = MI2Net_BraTS(num_cls=class_num).cuda()
    elif net_type == "MI2Net_MyoPS" and mode == "train":
        net = MI2Net_MyoPS(num_cls=class_num).cuda()
    elif net_type == "MI2Net_MyoPS" and mode == "test":
        net = MI2Net_MyoPS(num_cls=class_num).cuda()

    return net
