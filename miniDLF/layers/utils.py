import numpy as np

class ImgCol(object):
    def __init__(self, block_H, block_W, stride=1, padding=0):
        self.block_H = block_H
        self.block_W = block_W
        self.padding = padding
        self.stride = stride
        self.indices = None
        
    def get_img2col_indices(self, X_shape):
        N, iD, iH, iW = X_shape                
        oH = int((iH - self.block_H + 2*self.padding)/self.stride + 1) 
        oW = int((iW - self.block_W + 2*self.padding)/self.stride + 1)
        
        i0 = np.repeat(np.arange(self.block_H), self.block_W)
        i0 = np.tile(i0, iD)
        i1 = self.stride * np.repeat(np.arange(oH), oW)
        j0 = np.tile(np.arange(self.block_W), self.block_H * iD)
        j1 = self.stride * np.tile(np.arange(oW), oH)

        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(iD), self.block_H * self.block_W).reshape(-1, 1)
        return (k.astype(int), i.astype(int), j.astype(int))
    
    def img2col(self, X):        
        X_padded = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')            
        k, i, j = self.get_img2col_indices(X.shape)
        cols = X_padded[:, k, i, j]
        iD = X.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(self.block_H * self.block_W * iD, -1)
        return cols
    
    def col2img(self, cols, X_shape):    
        N, iD, iH, iW = X_shape
        H_padded, W_padded = iH + 2 * self.padding, iW + 2 * self.padding
        X_padded = np.zeros((N, iD, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_img2col_indices(X_shape)

        cols_reshaped = cols.reshape(self.block_H * self.block_W * iD, -1, N).transpose(2, 0, 1)
                
        np.add.at(X_padded, (slice(None), k, i, j), cols_reshaped)        
        if self.padding == 0:
            return X_padded
        return X_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]