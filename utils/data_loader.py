import numpy as np
from glob import glob
from utils.transformations import reshape4FakeCNN

def data_loader(root_dir, config, mode='train'):
    
    files = glob(root_dir + '*[0-9].npy')
    files_per_batch = int(config['batch_size']/config['f_size'])
    
    print(mode + ' on ' + str(len(files)) + ' samples')
    
    while True:
        
        if mode == 'train': 

            js = np.random.choice(range(len(files)), size=files_per_batch, replace=False)
            fs = [files[i] for i in js]
            X = [np.load(f) for f in fs]
            X = np.concatenate(X, axis=0)
            nrs = [f.split('/')[-1].split('.')[0] for f in fs]
            
            data_len = [np.load(root_dir + nr + '_len.npy') for nr in nrs]
            target_len = [np.load(root_dir + nr + '_target_len.npy') for nr in nrs]
            X_len = np.concatenate(data_len, axis=0)
            Y_len = np.concatenate(target_len, axis=0)
            
            Y_buff = [np.load(root_dir + nr + '_target.npy') for nr in nrs]
            lens = [y.shape[1] for y in Y_buff]
            max_seq_len = max(lens)
            N, T, F = X.shape
            Y = np.ones((N, max_seq_len), dtype=np.int) * 28

            idx = 0
            for y in Y_buff:
                Y[idx:idx+y.shape[0], :y.shape[1]] = y
                idx += y.shape[0]
                
            mask1 = Y_len != 0
#             mask2 = np.array([X[i].shape[0] >= Y_len[i] for i in range(X.shape[0])])
            mask = mask1 #+ mask2
                
            X, X_len = reshape4FakeCNN(X, X_len, 10, 2)
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
            
             # filter sequences that are after transformation for CNN shorter that target sequence!!!
            mask2 = X_len >= Y_len
            mask *= mask2

            X = X[mask]
            X_len = X_len[mask]
            Y = Y[mask]
            Y_len = Y_len[mask]

            yield X, X_len, Y, Y_len
                    
        if mode == 'val': 
            for f in files:
                
                nr = f.split('/')[-1].split('.')[0]

                X = np.load(f)
                X_len = np.load(root_dir + nr + '_len.npy')
                Y_len = np.load(root_dir + nr + '_target_len.npy')
                Y = np.load(root_dir + nr + '_target.npy')

                mask1 = Y_len != 0
                mask = mask1
                
                X, X_len = reshape4FakeCNN(X, X_len, 10, 2)
                X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
                
                # filter sequences that are after transformation for CNN shorter that target sequence!!!
                mask2 = X_len >= Y_len
                mask *= mask2
                
                X = X[mask]
                X_len = X_len[mask]
                Y = Y[mask]
                Y_len = Y_len[mask]
                
                yield X, X_len, Y, Y_len