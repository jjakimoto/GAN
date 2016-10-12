class DCGANConfig(object):
    device='/gpu:0'
    image_size=100
    is_crop=True
    sample_size = 64
    image_size =64 # shape == (image_size, image_size)
    k_h = 5
    k_w = 5
    d_h = 2
    d_w = 2
    z_dim=100
    # dimention of color
    c_dim=3
    dataset_name='sun'
    learning_rate=2.0e-4
    batch_size = 128
    n_epoch=1000
    checkpoint_dir='/path/to/your/checkpoint'
    save_img_dir='/path/to/your/save/directory'

