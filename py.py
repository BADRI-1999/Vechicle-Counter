# change the perspective of a line with a given angle opencv and focal length
# https://stackoverflow.com/questions/17087446/how-to-perform-perspective-transformation-in-opencv-python

def perspective_transform(line, angle, focal_length):
    """
    :param line: line to be transformed
    :param angle: angle of the line
    :param focal_length: focal length of the camera
    :return: transformed line
    """
    # get the coordinates of the line
    x1, y1, x2, y2 = line
    # get the angle of the line
    angle = angle * np.pi / 180
    # get the length of the line
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # get the coordinates of the transformed line
    x1_t = (x1 * np.cos(angle) - y1 * np.sin(angle)) / focal_length
    y1_t = (x1 * np.sin(angle) + y1 * np.cos(angle)) / focal_length
    x2_t = (x2 * np.cos(angle) - y2 * np.sin(angle)) / focal_length
    y2_t = (x2 * np.sin(angle) + y2 * np.cos(angle)) / focal_length
    # return the transformed line
    return [x1_t, y1_t, x2_t, y2_t]


# collision detection of a rectange with line of slope k
def collision_detection_rect(line, rect):
    x, y, w, h = rect
    x1, y1, x2, y2 = line
    if x1 > x and x1 < x + w and y1 > y and y1 < y + h:
        return True
    if x2 > x and x2 < x + w and y2 > y and y2 < y + h:
        return True
    return False
def collision_detection_line(line, rect):

def collision_detection(line, rect):
    x, y, w, h = rect
    x1, y1, x2, y2 = line
    if x1 > x and x1 < x + w and y1 > y and y1 < y + h:
        return True
    if x2 > x and x2 < x + w and y2 > y and y2 < y + h:
        return True
    return False