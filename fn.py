import cv2
import numpy as np


#take canny image
def HoughLines(edges, dTheta, threshold):
    imageShape = edges.shape
    #diameter of image
    Diameter = (imageShape[0]**2 + imageShape[1]**2)**0.5
    #range of rho values
    rhoRange = [i for i in range(int(Diameter)+1)]
    #range of thetas
    thetaRange = [dTheta*i for i in range(int(-np.pi/(2*dTheta)), int(np.pi/dTheta))]
    cosTheta = []
    sinTheta = []
    for theta in thetaRange:
        cosTheta.append(np.cos(theta))
        sinTheta.append(np.sin(theta))
    acc = np.zeros((len(rhoRange), len(thetaRange)))

    eds = []
    for (x,y), value in np.ndenumerate(edges):
        if value > 0:
            eds.append((x,y))
    #compute wanted rho and theta and store it in lines 
    for thetaIndex in range(len(thetaRange)):
        theta = thetaRange[thetaIndex]
        cos = cosTheta[thetaIndex]
        sin = sinTheta[thetaIndex]
        for x, y in eds:
            targetRho = x*cos + y*sin
            closestRhoIndex = int(round(targetRho))
            acc[closestRhoIndex, thetaIndex] += 1
            
    lines = []
    for (p,t), value in np.ndenumerate(acc):
        if value > threshold:
            lines.append((p,thetaRange[t]))

    return lines




def gaussian_smoothing(image):
                                
    gaussian_filter=np.array([[0.109,0.111,0.109],
                              [0.111,0.135,0.111],
                              [0.109,0.111,0.109]])
                                
    return cv2.filter2D(image,-1,gaussian_filter)  
        
def canny_edge_detection(gaussian_image):
    
    gaussian_image = gaussian_image.astype('uint8')

    otsu_threshold_val, ret_matrix = cv2.threshold(gaussian_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lower_threshold = otsu_threshold_val * 0.4
    upper_threshold = otsu_threshold_val * 1.3
    
    print(lower_threshold,upper_threshold)

    edges = cv2.Canny(gaussian_image, lower_threshold, upper_threshold)
    return edges


def HoughCircles(edges): 
    rows = edges.shape[0] 
    cols = edges.shape[1] 

    sinang = dict() 
    cosang = dict() 
    
    circles = []
    for angle in range(0,360): 
        sinang[angle] = np.sin(angle * np.pi/180) 
        cosang[angle] = np.cos(angle * np.pi/180)

    
    radius = [i for i in range(10,70)]
    threshold = 150 
    
    for r in radius:

        acc = np.zeroes((rows,cols))
        for x in range(rows): 
            for y in range(cols): 
                if edges[x][y] == 255:
                    for angle in range(0,360): 
                        #compute center of circles
                        a = x - round(r * cosang[angle]) 
                        b = y - round(r * sinang[angle]) 
                        acc[a][b] += 1
                             
        acc_max = np.amax(acc)
        
        if(acc_max > threshold):  
            acc[acc < threshold] = 0           
            # find the circles for this radius 
            for i in range(rows): 
                for j in range(cols): 
                    if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc[i][j] >= threshold):
                        avg_sum = np.float32((acc[i][j]+acc[i-1][j]+acc[i+1][j]+acc[i][j-1]+acc[i][j+1]+acc[i-1][j-1]+acc[i-1][j+1]+acc[i+1][j-1]+acc[i+1][j+1])/9) 
                        if(avg_sum >= 33):
                            circles.append((i,j,r))
                            acc[i:i+5,j:j+7] = 0
    return circles






















