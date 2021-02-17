class Point2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def inside_polygon(p, polygon):
    """
    This functions checks if a point is inside a certain lane (or area), a.k.a. polygon.
    (https://jsbsan.blogspot.com/2011/01/saber-si-un-punto-esta-dentro-o-fuera.html)

    Takes a point and a polygon and returns if it is inside (1) or outside(0).
    """
    counter = 0
    xinters = 0
    detection = False

    p1 = Point2D(0,0)
    p2 = Point2D(0,0)

    p1 = polygon[0] # First column = x coordinate. Second column = y coordinate

    for i in range(1,len(polygon)+1):
        p2 = polygon[i%len(polygon)]

        if (p.y > min(p1.y,p2.y)):
            if (p.y <= max(p1.y,p2.y)):
                if (p.x <= max(p1.x,p2.x)):
                    if (p1.y != p2.y):
                        xinters = (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x
                        if (p1.x == p2.x or p.x <= xinters):
                            counter += 1
        p1 = p2

    if (counter % 2 == 0):
        detection = False
    else:
        detection = True

    return detection

node_0_left = Point2D(6.79462553386422, 61.24215006927233)
node_1_left = Point2D(6.737679256245812, 52.24234080413561)
node_0_right = Point2D(3.294695602610389, 61.26429676910658)
node_1_right = Point2D(3.237749324991981, 52.26448750396986)

p = Point2D(5.088936447139291, 57.331452189028795)

polygon = [node_0_left,node_1_left,node_0_right,node_1_right]

ans = inside_polygon(p,polygon)

print("Is inside: ", ans)
