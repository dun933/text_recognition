import cairocffi as cairo, numpy as np 

def convert_cairo_format_to_rgb255base(cairo_format_rgb):
    """
    convert cairo format rgb(normed by 1.0) to 255 size normalized rgb value

    # output
    python list of rgb with each value in 255 base int value
    """
    rgb_list = list(cairo_format_rgb)
    rgb_list = list(map(lambda x: int(255*x), rgb_list))

    return rgb_list


def convert_rgb255base_to_cairo_format(rgb255base_color):
    """
    convert 255base rgb color tuple/list into cairo format(normalized by 1.0) tuple

    # output
    tuple
    """
    if isinstance(rgb255base_color, tuple):
        rgb255base_color = list(rgb255base_color)
    output_rgb = list(map(lambda x: float(x)/255 , rgb255base_color))

    return tuple(output_rgb)
    

def convert_surface_to_cvmat(surface, surface_width, surface_height):
    """
    works with FORMAT_RGB24
    it should work with FORMAT_ARGB32 too but the transparency will be ignored.

    # output
    cv2 mat in BGR format.
    """
    # print("surface format = {}".format(surface.get_format()))
    assert surface.get_format() == cairo.FORMAT_RGB24

    buf = surface.get_data()
    # print("format={}, given_surface_width:{}, actual_surface_width:{}, buf size: {}".format( surface.get_format(), surface_width, surface.get_width(), len(buf)))

    # if surface.get_width() != surface_width:
    #     surface.write_to_png("error.png")
        

    np_converted = np.ndarray(shape=(surface_height, surface_width),
                              dtype=np.uint32,
                              buffer=buf)

    # for r bitwise AND with 0x00ff0000
    r_filter = int("00ff0000", 16)
    g_filter = int("0000ff00", 16)
    b_filter = int("000000ff", 16)

    np_r_filtered = np.bitwise_and(np_converted, r_filter)
    np_r_shifted = np.right_shift(np_r_filtered, 16)
    np_r = np.expand_dims(np_r_shifted, axis=-1)

    np_g_filtered = np.bitwise_and(np_converted, g_filter)
    np_g = np.expand_dims(np.right_shift(np_g_filtered, 8), axis=-1)

    np_b = np.expand_dims(np.bitwise_and(np_converted, b_filter), axis=-1)

    # cv2 uses BGR format so concatenated in this order

    combined_32 = np.concatenate([np_b, np_g, np_r], axis=2)

    return combined_32.astype(np.uint8)


def create_rgb24_cairosurface_from_cv2mat(imgmat,):
    """
    imgmat: cv2 imgmat. in BGR format.
    opacity: in range 0~1
    background_color: the background color to use. tuple format. value range: 0~255
    """


    img_h, img_w, _ = imgmat.shape
    imgraw_bytearray = bytearray()

    for h_index in range(img_h):
        for w_index in range(img_w):
            pixel = imgmat[h_index, w_index,:]
    
            imgraw_bytearray.append(int(pixel[2]))
            imgraw_bytearray.append(int(pixel[1]))
            imgraw_bytearray.append(int(pixel[0]))


    # instead of using ndarray.tobytes(), the above will make sure that the rgb int values are concatenated.
    # in some cases, the imgmat values are not int, and it is float. in that case, ndarry.tobytes() will not give the size that we desire.
    # imgraw_bytearray = imgmat.tobytes()


    block_num = len(imgraw_bytearray) / 3
    block_num = int(block_num)

    stretched_bytes = bytearray()

    for index in range(block_num):
        start = 3*index
        
        first_index=start
        second_index = start+1
        third_index = start+2

        
        expanded_to_fourbytes = bytearray()

        # reversing the byte array order. since using smallendian
        expanded_to_fourbytes.append( imgraw_bytearray[third_index] )
        expanded_to_fourbytes.append( imgraw_bytearray[second_index])
        expanded_to_fourbytes.append( imgraw_bytearray[first_index])
        
        empty_byte = 0 

        expanded_to_fourbytes.append(empty_byte)
        
        stretched_bytes += expanded_to_fourbytes
        
    format = cairo.FORMAT_RGB24

    surface = cairo.ImageSurface.create_for_data(stretched_bytes, format, img_w, img_h)

    return surface