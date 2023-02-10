import cv2
import numpy as np

def distortPoints(img):
    h, w = img.shape[:2]
    x_list, y_list = np.meshgrid(np.arange(w), np.arange(h))
    pixel_locations = np.stack([x_list, y_list], axis=-1).reshape([h*w, 2])
    k1 = D[0]
    k2 = D[1]
    u0 = K[0][2]
    v0 = K[1][2]
    r = (pixel_locations[:, 0] - u0)**2 + (pixel_locations[:, 1] - v0)**2
    ud = (1 + k1*r + k2*(r**2)) * (pixel_locations[:, 0] - u0) + u0
    vd = (1 + k1*r + k2*(r**2)) * (pixel_locations[:, 1] - v0) + v0
    dist_px_locs = np.stack([ud, vd], axis=-1)
    intensity_vals = img[np.round(dist_px_locs[:, 1].astype(np.int16)),
                         np.round(dist_px_locs[:, 0].astype(np.int16))]
    result = intensity_vals.reshape(img.shape).astype(np.uint8)
    return result


def poseVectorToTransformationMatrix(pose_vec):
    w = pose_vec[:3]
    t = pose_vec[3:]
    t = t.reshape((3,1))
    
    theta = np.sqrt((w**2).sum())
    kx, ky, kz = w / theta
    R = np.array([
        [(kx**2)*(1-np.cos(theta)) + np.cos(theta), kx*ky*(1-np.cos(theta))-kz*np.sin(theta), kx*kz*(1-np.cos(theta))+ky*np.sin(theta)],
        [kx*ky*(1-np.cos(theta))+kz*np.sin(theta), (ky**2)*(1-np.cos(theta))+np.cos(theta), ky*kz*(1-np.cos(theta))-kx*np.sin(theta)],
        [kx*kz*(1-np.cos(theta))-ky*np.sin(theta), ky*kz*(1-np.cos(theta))+kx*np.sin(theta), (kz**2)*(1-np.cos(theta))+np.cos(theta)]
    ])
    
    T = np.append(R, t, axis=1)
    return T


def projectPoints(x, y, z, T):
    Pw = np.array([
        [x],
        [y],
        [z],
        [1]
    ])
    Pc = T @ Pw
    Ps = K @ Pc
    u, v, l = Ps
    u = u / l
    v = v / l
    return int(u), int(v)


def drawDots(img, Pw, T, showDisplay = False, color=(0,0,255)):
    if showDisplay:
        for pnt in Pw:
            u, v = projectPoints(pnt[0], pnt[1], 0, T)
            cv2.circle(img, (u, v), 2, color, 2)


def drawCube(img, x, y, T, height=-0.08, showDisplay=False, cubeSize = 0.08, color=(0,0,255)):
    if showDisplay:
        u0, v0 = projectPoints(x, y, 0, T)
        u1, v1 = projectPoints(x+cubeSize, y, 0, T)
        u2, v2 = projectPoints(x, y+cubeSize, 0, T)
        u3, v3 = projectPoints(x+cubeSize, y+cubeSize, 0, T)
        u4, v4 = projectPoints(x, y, height, T)
        u5, v5 = projectPoints(x+cubeSize, y, height, T)
        u6, v6 = projectPoints(x, y+cubeSize, height, T)
        u7, v7 = projectPoints(x+cubeSize, y+cubeSize, height, T)

        cv2.line(img, (u0,v0), (u1,v1), color, 3)
        cv2.line(img, (u0,v0), (u2,v2), color, 3)
        cv2.line(img, (u2,v2), (u3,v3), color, 3)
        cv2.line(img, (u1,v1), (u3,v3), color, 3)

        cv2.line(img, (u4,v4), (u5,v5), color, 3)
        cv2.line(img, (u4,v4), (u6,v6), color, 3)
        cv2.line(img, (u6,v6), (u7,v7), color, 3)
        cv2.line(img, (u5,v5), (u7,v7), color, 3)

        cv2.line(img, (u0,v0), (u4,v4), color, 3)
        cv2.line(img, (u1,v1), (u5,v5), color, 3)
        cv2.line(img, (u2,v2), (u6,v6), color, 3)
        cv2.line(img, (u3,v3), (u7,v7), color, 3)

def main():
    pose_vectors = np.loadtxt("data\poses.txt")
    global D 
    global K 
    D = np.loadtxt("data\D.txt")
    K = np.loadtxt("data\K.txt")

    Pw_x, Pw_y = np.meshgrid(np.arange(9)*0.04, np.arange(6)*0.04)
    Pw = np.stack([Pw_x, Pw_y], axis=-1).reshape(9*6, 2)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("my_cube.avi", fourcc, 30.0, (752, 480))

    index = 1
    stream = 0
    create_video = 1

    while True:
        img = cv2.imread("data\images\img_{0:04d}.jpg".format(index))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = distortPoints(img)
        result = cv2.cvtColor(result,  cv2.COLOR_GRAY2BGR)

        T = poseVectorToTransformationMatrix(pose_vectors[index-1])
        drawDots(result, Pw, T, showDisplay=True, color=(0,255,0))
        drawCube(result, 0, 0, T, height = -0.08, showDisplay=True, color=(255,0,0))

        index += 1
        
        if stream:
            cv2.imshow("Input", img)
            cv2.imshow("Output", result)
            cv2.waitKey(1)
        if create_video:
            video.write(result)
            print(f"Creating video, please wait... %{int((index/736)*100)}", end="\r")

if __name__ == "__main__":
    main()
