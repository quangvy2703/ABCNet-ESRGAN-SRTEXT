from PIL import Image
hr_shape = (480, 480)


def crop_into_boxes(img):
    coords = []
    hr_w, hr_h = hr_shape
    w, h = img.size
    for h_idx in range(0, h, hr_h):
        for w_idx in range(0, w, hr_w):
            if w_idx + hr_w > w:
                w_idx = w - hr_w
            if h_idx + hr_h > h:
                h_idx = h - hr_h
            box = (w_idx, h_idx, w_idx + hr_w, h_idx + hr_h)
            cropped_box = img.crop(box)
            coords.append(cropped_box)
            try:
                cropped_box.save("test/img_{}_{}.png".format(w_idx, h_idx))
            except:
                pass
    return coords

def merge_into_image(cropped_boxes, original_size):
    new_im = Image.new('RGB', original_size)

    x_offset = 0
    y_offset = 0
    for im in cropped_boxes:
        if x_offset + im.size[0] >= original_size[0]:
            x_offset = original_size[0] - im.size[0]
            new_im.paste(im, (x_offset, y_offset))
            y_offset += im.size[1]
            x_offset = 0
            continue
        if y_offset + im.size[1] > original_size[1]:
            y_offset = original_size[1] - im.size[1]
            # new_im.paste(im, (x_offset, y_offset))
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    new_im.save('test/combine.png')

img = Image.open("/home/ubuntu/demo/SRText/ABCNet-ESRGAN-SRTEXT/datasets/CTW1500/ctwtrain_text_image/0003.jpg")
original_size = img.size
shards = crop_into_boxes(img)
merge_into_image(shards, original_size)
