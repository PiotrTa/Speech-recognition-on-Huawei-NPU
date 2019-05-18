import numpy as np

def reshape4FakeCNN(x, input_len, kernel_w, stride_w):
    max_len = 0
    i = 0
    while i + kernel_w < x[0].shape[0]:
        i += stride_w
        max_len += 1
        
    max_len = ((max_len - 1) // 16 + 1) * 16
    
    x_new = np.zeros(shape=(x.shape[0], max_len, x.shape[2] * kernel_w))
    
    input_len_new = np.zeros(shape=(x.shape[0],))
    for ind in range(x.shape[0]):
        _ = x[ind]
        
        start = 0
        new_len = 0
        while start + kernel_w < input_len[ind]:
            arr = np.array([])
            for j in range(start, start + kernel_w):
                arr = np.concatenate((arr, _[j]))
            x_new[ind][new_len] = arr
            
            start += stride_w
            new_len += 1
        
        input_len_new[ind] = new_len
        
    return x_new, input_len_new