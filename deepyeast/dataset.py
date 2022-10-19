import os
import numpy as np
from PIL import Image
import hashlib
import tarfile

def validate_file(fpath, md5_hash):
    if not os.path.isfile(fpath):
        return False
    hasher = hashlib.md5()
    with open(fpath, 'rb') as ff:
        for chunk in iter(lambda: ff.read(1024 * 1024), b''):
            hasher.update(chunk)
    if hasher.hexdigest() != md5_hash:
        return False
    else:
        return True

def get_file(fname, url, md5_hash, extract=False):
    datadir = os.path.expanduser(os.path.join('~', '.deepyeast', 'datasets'))
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)
    
    download = False
    if os.path.exists(fpath):
        if not validate_file(fpath, md5_hash):
            print('File hash does not match the original md5 hash. Re-downloading data...')
            download = True
    else:
        download = True
        
    if download:
        print('Downloading data from ' + url)
        from six.moves import urllib
        urllib.request.urlretrieve(url, fpath)
        
    if extract:
        if fname.endswith(".tar.gz"):
            datadir = os.path.join(datadir, fname[:-7])
        if not os.path.exists(datadir):
            print('Extracting archive...')
            with tarfile.open(fpath, 'r:gz') as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, datadir)
        
    return fpath

def extract_filepaths_and_labels(path):
    with open(path, 'r') as ff:
        lines = ff.readlines()
        filepaths = [x.split(' ')[0] for x in lines]
        y = [int(x.strip().split(' ')[1]) for x in lines]
        y = np.array(y, dtype='uint8')
    return filepaths, y

def _load_data(category, split):
    if category == "main":
        datadir = os.path.expanduser(os.path.join('~', '.deepyeast', 'datasets', 'main'))
        doc_path = get_file('HOwt_doc.txt', 
                            'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_doc.txt', 
                            '33b7780020972e2da4f884c6b5a63b25')
        train_path = get_file('HOwt_train.txt', 
                              'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_train.txt', 
                              'b71eb4ff50f955adfa72048c6d8c0233')
        val_path = get_file('HOwt_val.txt', 
                            'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_val.txt', 
                            '2ac1d1874b89d6a1ad3d948d36c1e229')
        test_path = get_file('HOwt_test.txt', 
                             'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_test.txt', 
                             'c7958faa20232ff52fb196e754645bd1')
        data_path = get_file('main.tar.gz', 
                             'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/data/main.tar.gz', 
                             'f313fc0b8068941ab18ae65eb113afee',
                              extract=True)
    elif category == "transfer":
        datadir = os.path.expanduser(os.path.join('~', '.deepyeast', 'datasets', 'transfer'))
        doc_path = get_file('HOwt_transfer_doc.txt', 
                            'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_doc.txt', 
                            '699e4f5e390b8af78bbddae39f217745')
        train_path = get_file('HOwt_transfer_train.txt', 
                              'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_train.txt', 
                              '9dcbe909a81e3bdab6fa72504c4205a7')
        val_path = get_file('HOwt_transfer_val.txt', 
                            'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_val.txt', 
                            '32177364beaeabf54c4b4e589e2cb998')
        test_path = get_file('HOwt_transfer_test.txt', 
                             'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/image_prep/data/HOwt_transfer_test.txt', 
                             '3d5ff91995abfed6c0dd3b6e1727b6d2')
        data_path = get_file('transfer.tar.gz', 
                             'http://kodu.ut.ee/~leopoldp/2016_DeepYeast/data/transfer.tar.gz', 
                             '5dfc4baab5156e3fc6b5694bab96ac41',
                             extract=True)
    
    if split == 'train':
        filepaths, y = extract_filepaths_and_labels(train_path)
    elif split == 'val':
        filepaths, y = extract_filepaths_and_labels(val_path)
    elif split == 'test':
        filepaths, y = extract_filepaths_and_labels(test_path)
    filepaths = [os.path.join(datadir, x) for x in filepaths]
    
    print("Loading images...")    
    num_imgs = len(filepaths)
    x = np.empty((num_imgs, 64, 64, 2), dtype='uint8')
    for i in xrange(num_imgs):
        img = Image.open(filepaths[i])
        img = np.asarray(img, dtype='uint8')
        x[i] = img[:, :, :2]
        
    return x, y

def load_data(split):
    return _load_data('main', split)

def load_transfer_data(split):
    return _load_data('transfer', split)