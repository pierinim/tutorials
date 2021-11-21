import sys
import numpy as np
import cv2
import h5py
import os
from sklearn.utils import shuffle

addnoise=False
simple =True #single shape per figure
mixed  =False #multi shapes per figure
withBB =False  #one shape with bounding boxes

def background():
  #here one can add noise
  if addnoise:
    return np.array(np.random.rand(64,64,3)*20+100,np.uint8)
  else :
    return np.zeros((64,64,3), np.uint8)

def randomColor():
  return (int(np.random.rand()*128+128),int(np.random.rand()*128+128),int(np.random.rand()*128+128))

def drawCircle(c,x,y,r):
  img = background()
  cv2.circle(img,(x,y),r,c, -1)
  return img,x-r,y-r,x+r,y+r#return image and bounding box

def genCircle():
  return drawCircle(randomColor(),int(np.random.rand()*50)+10,int(np.random.rand()*50)+10,
                    int(np.random.rand()*6)+3)

def drawRectangle(c,x,y,w,h):
  img = background()
  cv2.rectangle(img,(x,y),((x+w),(y+h)), c, -1)
  return img,x,y,x+w,y+h #return image and bounding box

def genRectangle():
  return drawRectangle(randomColor(),int(np.random.rand()*40)+10,int(np.random.rand()*40)+10,
                       int(np.random.rand()*12)+5,int(np.random.rand()*12)+5)

def genN(f,i):
  img = np.zeros((64,64,3), np.uint8)
  for x in range(i):
    img+=f()[0] #discard bb info, take only image
  return img

def image_to_pointcloud(image, npixel):
    point_cloud = np.array([])
    for iX in range(image.shape[0]):
        for iY in range(image.shape[1]):
            pixel = np.array([iX, iY, image[iX,iY]])
            pixel = np.reshape(pixel, (1,3))
            point_cloud = np.concatenate((point_cloud, pixel), axis = 0) if point_cloud.size else pixel
    # select the darker npixel pixels
    point_cloud= point_cloud[point_cloud[:, -1].argsort()]
    point_cloud = point_cloud[-1*npixel:]
    point_cloud = np.reshape(point_cloud, (1, npixel, 3))
    return point_cloud

if __name__ == "__main__":
    nsamples=1000
    Npixel = 100
    split = int(0.7*nsamples)

    # Fully connected Graph Connection Matrix
    oneA = np.zeros((Npixel, Npixel))
    print(oneA.shape)
    np.fill_diagonal(oneA, 1)
    oneA = np.reshape(oneA, (1, Npixel, Npixel))
    print(oneA.shape)
    A = oneA
    for i in range(1,nsamples):
      A = np.concatenate((A, oneA), axis =0)
    A_train = A[:split,:,:]
    A_test = A[split:,:,:]
    del(A, oneA)
    print(A_train.shape, A_test.shape)
    
    nsamples = int(nsamples/2)
    
    images = np.reshape(genCircle()[0], (1,64, 64, 3))
    for i in range(1,nsamples):
      this_image = np.reshape(genCircle()[0], (1,64, 64, 3))
      images = np.concatenate([images, this_image], axis=0)
    print(images.shape)
 
    # B&W image: sum across the colors. We want to know if pixels are 0 or anything else
    images = np.sum(images, axis=-1)
    # Now introduce a random gray scale as G(0.9,0.1)
    images = np.random.normal(0.9,0.1,images.shape)*(images > 0)
    # generate noise images as G(0.1,0.1)
    noise = np.random.normal(0.7,0.1, images.shape)

    
    #noise = np.sum(noise, axis=-1)
    #noise = np.random.normal(0.7,0.1,noise.shape)*(noise >0)
    #noise = np.reshape(genRectangle()[0], (1,64, 64, 3))
    #for i in range(1,nsamples):
    #  this_image = np.reshape(genRectangle()[0], (1,64, 64, 3))
    #  noise = np.concatenate([noise, this_image], axis=0)
    #print(noise.shape)

    #
    images = np.concatenate([images, noise], axis=0)    
    del(noise)
    # now clip the values between 0 and 1
    images = np.minimum(np.ones(images.shape),images)
    images = np.maximum(np.zeros(images.shape),images)
    # create targets
    targets=np.concatenate([np.ones(nsamples), np.zeros(nsamples)], axis = 0)
    # shuffle
    images, targets = shuffle(images, targets)

    # from images to point cloud
    point_cloud = np.array([])
    for image in images:
        my_pc = image_to_pointcloud(image, Npixel)
        point_cloud = np.concatenate((point_cloud, my_pc), axis=0) if point_cloud.size else my_pc
    print(point_cloud.shape)
    del images

    # dataset
    Y_train = targets[:split]
    Y_test = targets[split:]
    X_train = point_cloud[:split,:,:]
    X_test = point_cloud[split:,:,:]
    print(Y_train.shape, Y_test.shape)
    print(X_train.shape, X_test.shape)
    del(targets, point_cloud)
    
    fOut = h5py.File(sys.argv[1], "w")
    fOut.create_dataset("X_train", data=X_train, compression="gzip")
    fOut.create_dataset("X_test", data=X_test, compression="gzip")
    fOut.create_dataset("Y_train", data=Y_train, compression="gzip")
    fOut.create_dataset("Y_test", data=Y_test, compression="gzip")
    fOut.create_dataset("A_train", data=A_train, compression="gzip")
    fOut.create_dataset("A_test", data=A_test, compression="gzip")
    fOut.close()

    
