from easydict import EasyDict as edict

_ = cfg = edict()

_.img_size = 112

_.train_list = ['train_good_7.txt']
_.test_list = ['test_good_7.txt']

_.wld = False

_.scale_list = [1, 2, 4]

_.weight_decay = 5e-4

_.loss = 'l1_loss'


# for prediction
_.overlap = 10
_.inf_scales = [1, 2, 4]
