class config:
    seg_type = 'binary' #binary | multiclass
    epochs = 100
    train_images = 'data/Train/images/*.png'
    train_mask = 'data/Train/masks/*.png'
    test_images = 'data/Test/images/*.png'
    test_masks = 'data/Test/masks/*.png'
    
    Multiclass_color_values = {
                0: [0, 0, 0],
                1: [255, 0, 0],
                2: [255, 215, 0],
                3: [0, 128, 0],
                4: [0, 0, 255],
                5: [139, 69, 19],
                6: [47, 79, 79],
                7: [25, 25, 112]
                }
    Binary_color_values = {
                0: [0, 0, 0],
                1: [255, 255, 255]
                }

    lr = 1e-4
    
