import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import glob, sys, os, inspect, ast
import pandas as pd 
from sklearn.utils.class_weight import compute_class_weight
import pylab as plt
import sys
sys.setrecursionlimit(10000)
print(f'current recursion limit set to: {sys.getrecursionlimit()}')

# unequal level of list depth/nesting
def flatten(S):
    """
    l = [2,[[1,2]],1]
    list(flatten(l))
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def interpolate_missing(A):
    # indx, A = interpolate_missing(Iq_M_WAXS[1])

    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    ok = ~ np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~ np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)

    return x, A

def get_dataframe_with_files_loc(file, sheet, BNL_dir, sub_dir):
    ### read dataframe
    df = pd.read_excel(file, sheet_name=sheet)
    df["File_Loc"] = pd.Series([], dtype=str)
    for idx, file in zip(df.index, df['File']):
        for file_search in glob.iglob(f'{BNL_dir}/**/*', recursive=True):
            if file_search.find(file) > -1 and f'/{sub_dir}/' in file_search:
                df.at[idx, 'File_Loc'] = file_search
                break
    return df

def train_val_split_dataset(func):
    def wrapper(*args, **kwargs):
        file, sheet, BNL_dir, sub_dir, val_files, test_files = args
        df = func(file, sheet, BNL_dir, sub_dir)

        def split_df(df, files):
            if files:
                val_indices = [df[df['File']==file].index.item() for file in files]
                df_val = df.iloc[val_indices]
                df_val.reset_index(inplace=True)
                df_train = df.drop(val_indices)
                df_train.reset_index(inplace=True)
                return df_train, df_val
            else:
                df_val = None
                return df, df_val

        df_train, df_val  = split_df(df, val_files)
        df_train, df_test = split_df(df_train, test_files)
        return df_train, df_val, df_test
        # df_valid = df.iloc[indices]
        
    return wrapper


def calculate_weights(pos, neg):

    total = pos + neg
    weights = [1,1]
    # total/2 --> (weight_for_0*neg +  weight_for_1* pos) =  total number of samples
    weights[0] = (1/neg)*(total/2.0)  # negative samples
    weights[1] = (1/pos)*(total/2.0)  # positive samples
    weights = np.array(weights)
    print(f'weight_for_0 : {weights[0]}, weight_for_1 : {weights[1]}, weights sum :  {weights[0]*neg +  weights[1]* pos}, total samoples : {total}')

    return weights

def Iq_scaling(Iq_input, Iq_scale, seek_mf, method = 'MSE'):
    ll, ul, inc = seek_mf
    mul_factor  = np.round(np.arange(ll, ul, inc),6)                                  # scaling factor lookup region
    err = {}; 
    err['MSE'] = {}
    err['NEG'] = {}
    for mf in mul_factor:
        if method == 'NEG':
            err['NEG'][mf] = Iq_input - mf*Iq_scale
        elif method == 'MSE':
            err['MSE'][mf] = np.mean(np.square(Iq_input - mf*Iq_scale ), axis=1);   # mean square error for scaling
    mf = np.zeros((Iq_input.shape[0],1))
    for idx in range(len(mf)):
        mf[idx] = sorted(err['MSE'].items(), key= lambda x: x[1][idx])[0][0]
    return mf


def get_intensities(df, column_name, label, BNL_dir, sub_dir):
    ### Variables for bkg column
    file_loc   = []
    frames_bkg = []
    Iq_values  = []
    labels_out = []
    frames_out = []
    files_out  = []
    indices    = df[column_name].dropna().index

    for idx in indices:

        if type(ast.literal_eval( df.iloc[idx][column_name]) ) == dict:
            # if not any([file in file_ for file_ in file_loc]):
            file   = list(ast.literal_eval( df.iloc[idx][column_name] ).keys())[0]
            if not any([file in file_ for file_ in file_loc]):   ## if already added to the stack do not add it again
                frames = flatten( list ( ast.literal_eval( df.iloc[idx][column_name]).values()) )

                for file_search in glob.iglob(f'{BNL_dir}/**/*', recursive=True):
                    if file_search.find(file) > -1 and f'/{sub_dir}/' in file_search:
                        file_loc.append(file_search)
                        break
                frames_bkg.append(frames)
        else:
            file = df.iloc[idx]['File']
            if not any([file in file_ for file_ in file_loc]):
                frames = flatten(list(ast.literal_eval( df.iloc[idx][column_name])))
                frames_bkg.append(frames)
                file_loc.append(df.iloc[idx]['File_Loc'])
    for file, frame in zip(file_loc, frames_bkg):
        df_temp     = pd.read_csv(file, delimiter=",", header=0)
        Iq_values.append( df_temp.iloc[frame].values.tolist() )
        labels_out += [label]*len(frame)
        frames_out += frame
        files_out  += [file]*len(frame)
    return np.vstack(Iq_values), np.array(labels_out), np.array(frames_out), np.array(files_out)   

class XrayData(Dataset):

    def __init__(self, df, column_names, BNL_dir, sub_dir, lidx=0, uidx=690, mica_sub=True, mica_Iq = None, tissue_sub=False, tissue_Iq = None, tissue_sub_indices = (360, 420), seek_mf = (-12,12,0.01), scaling=False):

        ### emptpy dataframe column is droped
        compute_column_names = []
        for column_name,label in column_names.items():
            if len(df[column_name].dropna()):
                compute_column_names.append((column_name,label))

        x_raw      = np.array([])
        labels_out = np.array([])
        frames_out = np.array([])
        files_out  = np.array([]) 

        for column_name,label in compute_column_names:
            Iq_values, labels, frames, files =  get_intensities(df, column_name, label, BNL_dir, sub_dir)
            print(f'{column_name} : contains {frames.size} samples')
            x_raw         = np.vstack([x_raw, Iq_values]) if x_raw.size else Iq_values
            labels_out    = np.hstack([labels_out, labels]) if labels_out.size else labels
            frames_out    = np.hstack([frames_out, frames]) if frames_out.size else frames
            files_out     = np.hstack([files_out, files]) if files_out.size else files

        if mica_sub:
            if isinstance(mica_Iq, np.ndarray):
                self.Iq_bkg = mica_Iq
            else:
                Iq_bkg , _, _, _ =  get_intensities(df, column_name='bkg_model', label=0, BNL_dir=BNL_dir, sub_dir=sub_dir)
                self.Iq_bkg = np.mean(Iq_bkg, axis=0)
                file = open("mica_bkg", "wb")              # Open a binary file in write mode
                np.save(file, self.Iq_bkg)                 # Save array to the file
                file.close()                               # Close the file
            x_raw -= self.Iq_bkg

            if tissue_sub:
                if isinstance(tissue_Iq, np.ndarray):
                    self.Iq_tissue = tissue_Iq
                else:
                    Iq_tissue , _, _, _ =  get_intensities(df, column_name='Tissue', label=1, BNL_dir=BNL_dir, sub_dir=sub_dir)
                    self.Iq_tissue = np.mean(Iq_tissue, axis=0, keepdims=True)
                    file = open("tissue_Iq", "wb")              # Open a binary file in write mode
                    np.save(file, self.Iq_tissue)               # Save array to the file
                    file.close()                                # Close the file
                
                tissue_lidx, tissue_uidx = tissue_sub_indices
                Iq_input = x_raw          [:,tissue_lidx : tissue_uidx]
                Iq_scale = self.Iq_tissue [:,tissue_lidx : tissue_uidx]
                self.mf = Iq_scaling(Iq_input, Iq_scale, seek_mf, method = 'MSE')
                x_raw -= self.mf*self.Iq_tissue

        x_raw = x_raw[:,lidx:uidx].astype(np.float32)
        _, x_raw   = interpolate_missing(x_raw)
        assert np.any(np.isnan(x_raw))==False, "X contains NaN values"
        self.x     = x_raw/np.max(x_raw,axis=1).reshape(-1,1) if scaling else x_raw

        self.y         = labels_out.astype(np.float32).reshape(-1,1)
        self.n_samples = len(self.y)
        self.frames    = frames_out
        self.files     = files_out

                 

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.n_samples   # len(dataset)


class triplet_data_generator(XrayData):
    def __init__(self, df, column_names, BNL_dir, sub_dir, lidx=0, uidx=690):
        super().__init__(df, column_names, BNL_dir, sub_dir, lidx, uidx)

        self.column_names = column_names
        self.ypos_idx = {}
        ### find indices for 0,1,2 from all y
        for (k,v) in self.column_names.items():
            self.ypos_idx[v] = np.where(self.y==v)[0]

        self.yneg_idx = {}
        for i in self.ypos_idx.keys():
            for j in self.ypos_idx.keys():
                if i!=j:
                    self.yneg_idx[i] = np.concatenate((self.yneg_idx.get(i, np.array([])) , self.ypos_idx[j]) , axis=None)
        
        # for (k,_) in self.y_idx.items():
        #     self.pos_labels[k] = self.y_idx[k]
 
            # self.neg_labels[k] = [_ for i in np.array(list(self.y_idx.keys())) != k]
        self._get_indices()
    
    def _get_indices(self):
        self.X_APN = np.full((1, self.x.shape[1]*3), np.nan)
        
        self.anc_pos_neg = []
        for idx in range(len(self.y)):
            idx_label = self.y[idx].item()
            anc_idx = idx
            pos_idx = idx if len(self.ypos_idx[ idx_label ]) ==1 else np.random.choice( np.delete(self.ypos_idx[idx_label],  self.ypos_idx[idx_label]==idx ))
            neg_idx = np.random.choice(self.yneg_idx[ idx_label ] ).astype(np.int64)
            self.anc_pos_neg.append( [anc_idx, pos_idx, neg_idx ])
            self.X_APN = np.vstack( (self.X_APN, np.hstack((self.x[anc_idx], self.x[pos_idx], self.x[neg_idx]))) )
        self.x = self.X_APN[1:].astype(np.float32)


def get_dataloaders_random_split(Excel_File, sheet, BNL_dir, sub_dir, column_names, lidx=0, uidx=690):

    batch_size  = 4096
    train_split = 0.8
    shuffle_dataset = True
    random_seed = 42

    # df = get_dataframe_with_files_loc(Excel_File, sheet, BNL_dir, sub_dir)
    # dataset = XrayData(df, column_names, BNL_dir, sub_dir, lidx=lidx, uidx=uidx)

    dataset = DatasetTester()

    # Creating data indices for training and validation splits:
    indices = np.arange(0,len(dataset))
    split   = int(train_split*len(dataset))

    print(f'train size: {split} test size : {len(dataset) - split}')

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Calculate weights
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset.y[train_indices]), y=dataset.y[train_indices].flatten().tolist())
    weights = torch.tensor(weights, dtype=torch.float32)
    print([f'weight {i}:{weight} ' for i,weight in enumerate(weights)])

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_loader   = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    print("training_loader size : {} ; validation_loader size : {}".format(len(training_loader), len(validation_loader)))

    return weights, training_loader, validation_loader

def get_sonar_dataloaders(sonar_file):

    class DatasetTester(nn.Module):
        def __init__(self):

            from sklearn.preprocessing import LabelEncoder

            # Read data
            data = pd.read_csv(sonar_file, header=None)
            self.X = data.iloc[:, 0:60].values
            y = data.iloc[:, 60]
             
            # Binary encoding of labels
            encoder = LabelEncoder()
            encoder.fit(y)
            self.y = encoder.transform(y)

            ### convert it to tensor
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32)

        def __getitem__(self,  index):
            return self.X[index], self.y[index]

        def __len__(self):
            return len(self.y)

    batch_size  = 4096
    train_split = .8
    shuffle_dataset = True
    random_seed = 42

    # df = get_dataframe_with_files_loc(Excel_File, sheet, BNL_dir, sub_dir)
    # dataset = XrayData(df, column_names, BNL_dir, sub_dir, lidx=lidx, uidx=uidx)

    dataset = DatasetTester()

    # Creating data indices for training and validation splits:
    indices = np.arange(0,len(dataset))
    split   = int(train_split*len(dataset))

    print(f'train size: {split} test size : {len(dataset) - split}')

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Calculate weights
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset.y[train_indices]), y=dataset.y[train_indices].flatten().tolist())
    weights = torch.tensor(weights, dtype=torch.float32)
    print([f'weight {i}:{weight} ' for i,weight in enumerate(weights)])

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_loader   = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    print("training_loader size : {} ; validation_loader size : {}".format(len(training_loader), len(validation_loader)))

    return weights, training_loader, validation_loader    

def get_dataloaders_fixed_val_test_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, test_files, lidx=0, uidx=690, mica_sub=True, scaling=False):
    batch_size  = 4096

    split_dataset             = train_val_split_dataset(get_dataframe_with_files_loc)
    df_train, df_val, df_test = split_dataset(Excel_File, sheet, BNL_dir, sub_dir, val_files, test_files)

    print('Setting Training Dataset ...')
    dataset_train = XrayData(df_train, column_names, BNL_dir, sub_dir, lidx=lidx, uidx=uidx, mica_sub=mica_sub, scaling=scaling)
    print('Setting Validation Dataset ...')
    dataset_val   = XrayData(df_val, column_names, BNL_dir, sub_dir, lidx=lidx, uidx=uidx, mica_sub=mica_sub, scaling=scaling) if isinstance(df_val,pd.core.frame.DataFrame) else None
    print('Setting Testing Dataset ...')
    dataset_test  = XrayData(df_test, column_names, BNL_dir, sub_dir, lidx=lidx, uidx=uidx, mica_sub=mica_sub, scaling=scaling) if isinstance(df_test, pd.core.frame.DataFrame) else None

    # Calculate weights on training set
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset_train.y), y=dataset_train.y.ravel())
    weights = torch.tensor(weights, dtype=torch.float32)
    print([f'weight {i} : {weight:0.2f} ' for i,weight in enumerate(weights)])

    training_loader   = DataLoader(dataset_train, batch_size=batch_size, num_workers=0)
    validation_loader = DataLoader(dataset_val,   batch_size=batch_size, num_workers=0) if dataset_val else None
    testing_loader    = DataLoader(dataset_test,  batch_size=batch_size, num_workers=0) if dataset_test else None
    print("training_loader size : {} ; validation_loader size : {} ; testing_loader size : {}".format(len(training_loader), len(validation_loader) if validation_loader else None, len(testing_loader) if testing_loader else None))

    return weights, training_loader, validation_loader, testing_loader

if __name__ == '__main__':
    ### specs : names - file , columns

    lidx = 250
    uidx = 340
    column_names= {"Neuritic_Plaque":1., "Diffuse_Plaque":1., "Neurofibrillary_Tangle_(tau)":1., "Tissue":0.}  # "Tau":1., "Neuritic_Plaque":1., "Diffuse_Plaque":1., "Neurofibrillary_Tangle_(tau)":1.,   # {"Diffuse_Plaque":0., "Neurofibrillary_Tangle_(tau)":1. , "Tau":2. ,"Neuritic_Plaque":3., "Tissue":4., "bkg":5.0 }
    Excel_File  = "/Users/bashit.a/Documents/Alzheimer/Mar-2023/Mar-2023-Samples-updated.xlsx"   # "/home/bashit.a/Codes/ML/Mar-2023-Samples.xlsx"   "/Users/bashit.a/Documents/Alzheimer/Mar-2023/Mar-2023-Samples.xlsx"    sheet       = 'Mar-2023-Samples'
    BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023'    # '/Volumes/HDD/BNL-Data/Mar-2023'         '/scratch/bashit.a/BNL-Data/Mar-2023'
    sub_dir     = "CSV_Conv-8-point"  # CSV_Conv-8-point  CSV
    val_files   = ["1898_CING-roi0_0_0_masked_intp.h5"]  # ["1948 V1-roi0_0_0_masked.h5"] # None # ["1948 V1-roi0_0_0_masked.h5"] # None ["1948_HIPPO-roi1_0_0_masked_intp.h5", "2428-roi1_0_0_masked_intp.h5"]
    test_files  = ["1948 V1-roi0_0_0_masked.h5"]  # None ["1948 V1-roi0_0_0_masked.h5"]
    sheet       = 'Mar-2023-Samples'
    mica_sub = True
    scaling  = False
    tissue_sub = False
    # print('Random Split Dataset -->')
    # weights, training_loader, validation_loader = get_dataloaders_random_split(Excel_File, sheet, BNL_dir, sub_dir, column_names, lidx=lidx, uidx=uidx)
    # train_dataiter = iter(training_loader)
    # train_data, train_label = next(train_dataiter)
    # print(type(train_data), train_data.shape, len(training_loader))

    # val_dataiter = iter(validation_loader)
    # val_data, val_label = next(val_dataiter)
    # print(type(val_data), val_data.shape, len(validation_loader))

    print('Fixed Validation Dataset -->')
    weights, training_loader, validation_loader, testing_loader = get_dataloaders_fixed_val_test_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, test_files, lidx=lidx, uidx=uidx, mica_sub=mica_sub, scaling=scaling)
    train_dataiter = iter(training_loader)
    train_data, train_label = next(train_dataiter)
    print(type(train_data), train_data.dtype, train_data.shape, train_label.shape, train_label.dtype, len(training_loader))
    plt.figure()
    plt.plot(train_data.detach().numpy().T)
    plt.show()

    if validation_loader:
        val_dataiter = iter(validation_loader)
        val_data, val_label = next(val_dataiter)
        print(type(val_data), val_data.dtype, val_data.shape, val_label.shape, val_label.dtype, len(validation_loader))

    if testing_loader:
        test_dataiter = iter(testing_loader)
        test_data, test_label = next(test_dataiter)
        print(type(test_data), test_data.dtype, test_data.shape, test_label.shape, test_label.dtype, len(testing_loader))
    # print(validation_loader.dataset.frames)

    for batch_idx, inputs in enumerate(training_loader):
        X = inputs[0]
        y = inputs[1]
        # print(type(X), type(y), X.shape, y.shape)

