from miniDLF.models import Sequential

class Autoencoder(object):
    def __init__(self, enc_model, dec_model, loss, optimizer):
        self.encoder = enc_model
        self.decoder = dec_model                
        self.loss = loss
        self.optimizer = optimizer
        
        self.autoencoder = self.__build_autoencoder__()        
    
    def set_encoder(self, model):
        self.encoder = model        
    
    def set_decoder(self, model):
        self.decoder = model
    
    def __build_autoencoder__(self):
        m = Sequential()
        for i in range(self.encoder.nLayers):
            m.add(self.encoder.layers[i])
        for i in range(self.decoder.nLayers):
            m.add(self.decoder.layers[i])
        m.compile(self.loss, self.optimizer)
        return m  

    def summary(self):
        self.autoencoder.summary()
        
    def fit(self, 
            dataset,  #   
            epochs,       # number of trials           
            minibatch_size = 10,
            early_stop_after = 5):
        
        self.autoencoder.fit(dataset=dataset, 
                             epochs=epochs, 
                             minibatch_size=minibatch_size, 
                             early_stop_after=early_stop_after, 
                             regression=True)
            
    def autoencode(self, X):
        a = X
        for i in range(0, self.autoencoder.nLayers): 
            a = self.autoencoder.layers[i].forward(a)            
        return a
    
    def encode(self, X):
        a = X
        for i in range(0, self.encoder.nLayers): 
            a = self.autoencoder.layers[i].forward(a)            
        return a
    
    def decode(self, X):
        a = X
        for i in range(self.encoder.nLayers, self.autoencoder.nLayers): 
            a = self.autoencoder.layers[i].forward(a)            
        return a