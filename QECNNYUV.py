import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from YUV_RGB import yuv2rgb
import tensorflow
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2D,UpSampling2D
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,Activation, concatenate
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from tensorflow.keras.callbacks import ModelCheckpoint

#Frame size of training data
w=480
h=320
#patch size and petch step for training
patchsize = 40
patchstep = 20

#test folders for raw and compressed in yuv and png formats
testfolderRawYuv = 'testrawyuv/'
testfolderRawPng = 'testrawpng/'
testfolderCompYuv = 'testcompyuv/'
testfolderCompPng = 'testcomppng/'

#train folders for raw and compressed in yuv and png formats
trainfolderRawYuv = 'trainrawyuv/'
trainfolderRawPng = 'trainrawpng/'
trainfolderCompYuv = 'traincompyuv/'
trainfolderCompPng = 'traincomppng/'

def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def yuv2rgb (Y,U,V,fw,fh):
    U_new = cv2.resize(U, (fw, fh),cv2.INTER_CUBIC)
    V_new = cv2.resize(V, (fw, fh), cv2.INTER_CUBIC)
    U = U_new
    V = V_new
    Y = Y
    rf = Y + 1.4075 * (V - 128.0)
    gf = Y - 0.3455 * (U - 128.0) - 0.7169 * (V - 128.0)
    bf = Y + 1.7790 * (U - 128.0)

    for m in range(fh):
        for n in range(fw):
            if (rf[m, n] > 255):
                rf[m, n] = 255
            if (gf[m, n] > 255):
                gf[m, n] = 255
            if (bf[m, n] > 255):
                bf[m, n] = 255
            if (rf[m, n] < 0):
                rf[m, n] = 0
            if (gf[m, n] < 0):
                gf[m, n] = 0
            if (bf[m, n] < 0):
                bf[m, n] = 0
    r = rf
    g = gf
    b = bf
    return r, g, b

def FromFolderYuvToFolderPNG (folderyuv,folderpng,fw,fh):
    dir_list = os.listdir(folderpng)
    for name in dir_list:
        os.remove(folderpng+name)
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')
    #list of patch left-top coordinates
    numdx = (fw-patchsize)//patchstep
    dx = np.zeros(numdx)
    numdy = (fh - patchsize) // patchstep
    dy = np.zeros(numdy)
    for i in range(numdx):
        dx[i]=i*patchstep
    for i in range(numdy):
        dy[i]=i*patchstep
    dx = dx.astype(int)
    dy = dy.astype(int)
    Im = np.zeros((patchsize, patchsize,3))
    dir_list = os.listdir(folderyuv)
    pngframenum = 0
    for name in dir_list:
        fullname = folderyuv + name
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size= fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2*size)//(fw*fh*3)
            frames=100
            print(fullname,frames)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                r,g,b = yuv2rgb (Y,U,V,fw,fh)
                for i in range(numdx):
                    for j in range(numdy):
                        Im[:, :, 0] = b[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        Im[:, :, 1] = g[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        Im[:, :, 2] = r[dy[j]:dy[j]+patchsize,dx[i]:dx[i]+patchsize]
                        pngfilename = "%s/%i.png" % (folderpng,pngframenum)
                        cv2.imwrite(pngfilename, Im)
                        pngframenum = pngframenum + 1
            fp.close()
    return (pngframenum-1)

#reads all images from folder and puts them into x array
def LoadImagesFromFolder (foldername):
    dir_list = os.listdir(foldername)
    N = 0
    Nmax = 0
    for name in dir_list:
        fullname = foldername + name
        Nmax = Nmax + 1

    x = np.zeros([Nmax, patchsize, patchsize, 3])
    N = 0
    for name in dir_list:
        fullname = foldername + name
        I1 = cv2.imread(fullname)
        x[N, :, :, 0] = I1[:, :, 2]
        x[N, :, :, 1] = I1[:, :, 1]
        x[N, :, :, 2] = I1[:, :, 0]
        N = N + 1
    return x

def psnr(y_true, y_pred):
    # Вычисляем MSE (Mean Squared Error)
    mse = tensorflow.reduce_mean(tensorflow.square(y_true - y_pred))
    # Задаем максимальное значение пикселя (например, для изображений с нормализацией от 0 до 1 это 1.0)
    max_pixel_value = 1.0
    # Вычисляем PSNR
    psnr = 10.0 * tensorflow.math.log((max_pixel_value ** 2) / mse) / tensorflow.math.log(10.0)
    return psnr


def EnhancerModel(fw, fh):
    comp_tensor = layers.Input(shape=(fh, fw, 3))

    # Residual Dense Block (RDB)
    def RDB(input_tensor, filters):
        x = layers.Conv2D(filters, kernel_size=3, padding="same", activation='relu')(input_tensor)
        x = layers.Conv2D(filters, kernel_size=3, padding="same", activation='relu')(x)
        return layers.Add()([input_tensor, x])

    # Initial convolution
    conv_1 = layers.Conv2D(filters=128, kernel_size=[9, 9], padding="same", activation='relu', name='conv_1')(
        comp_tensor)

    # Add RDBs
    rdb_1 = RDB(conv_1, 128)
    rdb_2 = RDB(rdb_1, 128)
    rdb_3 = RDB(rdb_2, 128)

    # Attention Mechanism
    attention = layers.Conv2D(128, kernel_size=1, activation="sigmoid")(rdb_3)
    enhanced_features = layers.Multiply()([rdb_3, attention])

    # Final layers
    conv_2 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation='relu', name='conv_2')(
        enhanced_features)
    conv_out = layers.Conv2D(filters=3, kernel_size=[3, 3], padding="same", name='conv_out')(conv_2)

    # Residual connection
    output_tensor = comp_tensor + conv_out

    enhancer = Model(inputs=comp_tensor, outputs=output_tensor)
    return enhancer


def TrainImageEnhancementModel(folderRaw, folderComp, folderRawVal, folderCompVal):
    print('Loading raw train images...')
    Xraw = LoadImagesFromFolder(folderRaw)
    print('Loading compressed train images...')
    Xcomp = LoadImagesFromFolder(folderComp)
    Xraw = Xraw / 255.0
    Xcomp = Xcomp / 255.0

    print('Loading raw validation images...')
    XrawVal = LoadImagesFromFolder(folderRawVal)
    print('Loading compressed validation images...')
    XcompVal = LoadImagesFromFolder(folderCompVal)
    XrawVal = XrawVal / 255.0
    XcompVal = XcompVal / 255.0

    enhancer = EnhancerModel(patchsize, patchsize)

    # Advanced learning rate schedule
    learning_rate_schedule = tensorflow.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001, decay_steps=10000, alpha=0.01)

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # Perceptual loss
    vgg = tensorflow.keras.applications.VGG19(include_top=False, weights="imagenet",
                                              input_shape=(patchsize, patchsize, 3))
    vgg.trainable = False

    def perceptual_loss(y_true, y_pred):
        true_features = vgg(y_true)
        pred_features = vgg(y_pred)
        return tensorflow.reduce_mean(tensorflow.square(true_features - pred_features))

    # Compile model
    enhancer.compile(loss=perceptual_loss, optimizer=optimizer, metrics=[psnr])

    checkpoint_filepath = 'best_model.weights.h5'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_psnr',
        save_best_only=True,
        save_weights_only=True,
        mode='max',  # Maximize PSNR
        verbose=1
    )

    # Callback for intermediate saving every 10 epochs
    class IntermediateCheckpoint(tensorflow.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:  # Save every 10 epochs
                filepath = f'model_epoch_{epoch + 1}.weights.h5'
                self.model.save_weights(filepath)
                print(f'Intermediate checkpoint saved at epoch {epoch + 1}: {filepath}')

    NumEpochs = 200
    hist = enhancer.fit(
        Xcomp, Xraw,
        epochs=NumEpochs,
        batch_size=64,
        verbose=1,
        validation_data=(XcompVal, XrawVal),
        callbacks=[checkpoint_callback, IntermediateCheckpoint()]
    )

    enhancer.save_weights('enhancer.weights.h5')
    return enhancer


def InferenceImageEnhancementModel(fw,fh):
    enhancer = EnhancerModel(fw,fh)
    enhancer.compile(loss='mean_squared_error',optimizer='Adam',metrics=[psnr])
    enhancer.load_weights('intermediate_model_epoch_10.weights.h5')

    return enhancer


def GetRGBFrame (folderyuv,VideoNumber,FrameNumber,fw,fh):
    fwuv = fw // 2
    fhuv = fh // 2
    Y = np.zeros((fh, fw), np.uint8, 'C')
    U = np.zeros((fhuv, fwuv), np.uint8, 'C')
    V = np.zeros((fhuv, fwuv), np.uint8, 'C')

    dir_list = os.listdir(folderyuv)
    v=0
    for name in dir_list:
        fullname = folderyuv + name
        if v!=VideoNumber:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            fp = open(fullname, 'rb')
            frames = (2 * size) // (fw * fh * 3)
            for f in range(frames):
                for m in range(fh):
                    for n in range(fw):
                        Y[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        U[m, n] = ord(fp.read(1))
                for m in range(fhuv):
                    for n in range(fwuv):
                        V[m, n] = ord(fp.read(1))
                if f==FrameNumber:
                    r, g, b = yuv2rgb(Y, U, V, fw, fh)
                    return r,g,b

def GetEngancedRGB (RGBin,fw,fh):
    RGBin = np.expand_dims(RGBin, axis=0)
    EnhancedPatches = enhancer.predict(RGBin)
    EnhancedPatches=np.squeeze(EnhancedPatches, axis=0)
    return EnhancedPatches

def ShowOneFrameEnhancement(folderyuvraw,foldercomp,VideoIndex,FrameIndex):
    r1, g1, b1 = GetRGBFrame(folderyuvraw,VideoIndex, FrameIndex, w, h)
    RGBRAW = np.zeros((h, w, 3))
    RGBRAW[:, :, 0] = r1
    RGBRAW[:, :, 1] = g1
    RGBRAW[:, :, 2] = b1

    r2, g2, b2 = GetRGBFrame(foldercomp, VideoIndex, FrameIndex, w, h)
    RGBCOMP = np.zeros((h, w, 3))
    RGBCOMP[:, :, 0] = r2
    RGBCOMP[:, :, 1] = g2
    RGBCOMP[:, :, 2] = b2

    RGBENH = GetEngancedRGB(RGBCOMP, w, h)

    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.subplot(1, 3, 1)
    plt.imshow(RGBRAW / 255.0)
    psnr1 = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
    psnr2 = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)

    tit = "%.2f, %.2f" % (psnr1, psnr2)
    plt.title(tit)

    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.subplot(1, 3, 2)

    plt.imshow(RGBCOMP / 255.0)

    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(RGBENH / 255.0)
    plt.show()

def ShowFramePSNRPerformance (folderyuv,foldercomp,VideoIndex,framesmax,fw,fh):
    RGBRAW = np.zeros((h, w, 3))
    RGBCOMP = np.zeros((h, w, 3))
    dir_list = os.listdir(folderyuv)
    v = 0
    for name in dir_list:
        fullname = folderyuv + name
        print(name)
        if v != VideoIndex:
            v = v + 1
            continue
        if fullname.endswith('.yuv'):
            fp = open(fullname, 'rb')
            fp.seek(0, 2)  # move the cursor to the end of the file
            size = fp.tell()
            fp.close()
            frames = (2 * size) // (fw * fh * 3)
            if frames>framesmax:
                frames = framesmax

            PSNRCOMP = np.zeros((frames))
            PSNRENH = np.zeros((frames))
            for f in range(frames):
                print(f,frames)
                r, g, b = GetRGBFrame(folderyuv, VideoIndex, f, w, h)
                RGBRAW[:, :, 0] = r
                RGBRAW[:, :, 1] = g
                RGBRAW[:, :, 2] = b
                r, g, b = GetRGBFrame(foldercomp, VideoIndex, f, w, h)
                RGBCOMP[:, :, 0] = r
                RGBCOMP[:, :, 1] = g
                RGBCOMP[:, :, 2] = b
                PSNRCOMP[f] = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
                RGBENH = GetEngancedRGB(RGBCOMP, w, h)
                PSNRENH[f] = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)
        break

    ind = np.argsort(PSNRCOMP)

    plt.plot(PSNRCOMP[ind], label='Compressed')
    plt.plot(PSNRENH[ind], label='Enhanced')
    plt.xlabel('Frame index')
    plt.ylabel('PSNR, dB')
    plt.grid()
    plt.legend()
    tit = "%s PSNR = [%.2f, %.2f] dB" % (name,np.mean(PSNRCOMP), np.mean(PSNRENH))
    plt.title(tit)
    plt.show()



TrainMode = 1
PrepareDataSetFromYUV = 0

if TrainMode==1:
    if PrepareDataSetFromYUV==0:
        FromFolderYuvToFolderPNG (testfolderRawYuv,testfolderRawPng,w,h)
        FromFolderYuvToFolderPNG (testfolderCompYuv,testfolderCompPng,w,h)
        FromFolderYuvToFolderPNG (trainfolderRawYuv,trainfolderRawPng,w,h)
        FromFolderYuvToFolderPNG (trainfolderCompYuv,trainfolderCompPng,w,h)
    TrainImageEnhancementModel(trainfolderRawPng,trainfolderCompPng,testfolderRawPng,testfolderCompPng)



if 1:
    enhancer = InferenceImageEnhancementModel (w,h)
    #ShowOneFrameEnhancement(trainfolderRawYuv,trainfolderCompYuv,0,0)
    #ShowOneFrameEnhancement(testfolderRawYuv,testfolderCompYuv,0,0)
    #ShowOneFrameEnhancement(trainfolderRawYuv, trainfolderCompYuv, 0, 1)
    #ShowOneFrameEnhancement(testfolderRawYuv, testfolderCompYuv, 0, 1)
    ShowFramePSNRPerformance (trainfolderRawYuv,trainfolderCompYuv,0,100,w,h)
    ShowFramePSNRPerformance (testfolderRawYuv,testfolderCompYuv,0,100,w,h)
