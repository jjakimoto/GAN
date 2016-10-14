class DCGANConfig(object):
    device='/gpu:0'
    sample_size = 64
    image_size =64 # shape == (image_size, image_size)
    k_h = 5
    k_w = 5
    d_h = 2
    d_w = 2
    z_dim=100
    # if is_color is true each data have three dimentio
    is_color = True
    # dimention of color
    c_dim=3
    dataset_name='sun'
    learning_rate=2.0e-4
    batch_size = 128
    n_epoch=1000
    checkpoint_dir='/path/to/your/checkpoint'
    save_img_dir='/path/to/your/save/directory'

class mnistConfig(obejct):
    device='/gpu:0'
    sample_size = 64
    image_size =28 # shape == (image_size, image_size)
    k_h = 5
    k_w = 5
    d_h = 2
    d_w = 2
    z_dim=100
    # if is_color is False c_dim is not necessary to specify
    is_color = False
    dataset_name='sun'
    learning_rate=2.0e-4
    batch_size = 128
    n_epoch=1000
    checkpoint_dir='/path/to/your/checkpoint'
    save_img_dir='/path/to/your/save/directory'
