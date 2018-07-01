from keras.models import Model
import numpy as np

CLASS_INDEX = ['cell periphery',
               'cytoplasm',
               'endosome',
               'er',
               'golgi',
               'mitochondrion',
               'nuclear periphery',
               'nucleolus',
               'nucleus',
               'peroxisome',
               'spindle pole',
               'vacuole']

def preprocess_input(x):
    x = x.astype(np.float32)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def decode_predictions(preds):
    results = []
    for pred in preds:
        result = [(CLASS_INDEX[i], pred[i]) for i in range(len(pred))]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def create_feature_extractor(base_model, layer_name):
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer_name).output)
    return model

def mixup_generator(x, y, batch_size, alpha=0.2, dataaug=None):
    num_iters = x.shape[0]//batch_size
    while 1:
        idx_shf1 = np.random.permutation(x.shape[0])
        idx_shf2 = np.random.permutation(x.shape[0])
        for i in range(num_iters):
            idx1 = idx_shf1[i*batch_size:(i+1)*batch_size]
            idx2 = idx_shf2[i*batch_size:(i+1)*batch_size]
            coef = np.random.beta(alpha, alpha, batch_size)
            coef = coef.reshape(batch_size, 1, 1, 1)
            x1 = x[idx1]
            x2 = x[idx2]
            if dataaug:
                for j in range(batch_size):
                    x1[j] = dataaug.random_transform(x1[j])
                    x1[j] = dataaug.standardize(x1[j])
                    x2[j] = dataaug.random_transform(x2[j])
                    x2[j] = dataaug.standardize(x2[j])
            batch_x = coef * x1 + (1 - coef) * x2
            coef = coef.reshape(batch_size, 1)
            batch_y = coef * y[idx1] + (1 - coef) * y[idx2]
            yield batch_x, batch_y
