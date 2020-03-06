from miniDLF.models import Sequential

class Autoencoder(object):
    def __init__(self, enc_model, dec_model, loss, optimizer):
        self.encoder = enc_model
        self.decoder = dec_model                
        self.loss = loss
        self.optimizer = optimizer
        
        self.autoencoder = self.build_autoencoder()        
    
    def set_encoder(self, model):
        self.encoder = model        
    
    def set_decoder(self, model):
        self.decoder = model
    
    def build_autoencoder(self):
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
              accuracy_threshold = 1.0, 
              minibatch_size = 10,
              early_stop_after = 5):
        
        self.autoencoder.fit(dataset, epochs, accuracy_threshold, 
                             minibatch_size, early_stop_after, True)
            
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