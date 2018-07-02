from easydict import EasyDict as edict

_ = cfg = edict()

_.img_size = 112

_.train_list = ['train_good_1.txt']
_.test_list = ['test_good_1.txt']

_.wld = True

_.scale_list = [1, 2, 4]

_.weight_decay = 5e-4
