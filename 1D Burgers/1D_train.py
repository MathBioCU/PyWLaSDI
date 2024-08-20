import sys,time,os
import pickle

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

sys.path.append("../..")
#import modAutoEncoder as autoencoder
import modLaSDIUtils as utils
import torch


from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from torch.optim import lr_scheduler
import subprocess as sp
import Autoencoder_pytorch
import copy
#torch.cuda.empty_cache()
# Set print option
np.set_printoptions(threshold=sys.maxsize)

# Choose device that is not being used
gpu_ids = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

def getDevice():
#evice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda") #torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
# set device
device = getDevice()
print("Using device:", device, '\n')



# set batch_size, number of epochs, paitience for early stop
batch_size = 1001
num_epochs = 10000
num_epochs_print = num_epochs//100
early_stop_patience = num_epochs//50


LS_dim = 5

print('')
print("Beginning Latent Space Dimension: {}".format(LS_dim))
print('')

nt = 1000


# load snapshot
file_name_snapshot='./data/local16_10pc_noise.p'
snapshot = pickle.load(open(file_name_snapshot,'rb'))


# number of data points
data = []
for i in range(16):
    snapshot_data = snapshot['data'][i]['x'].astype('float32')
    data.append(snapshot_data)
data = np.vstack(data)


ndata = data.shape[0]
orig_data = data
# check shapes of snapshot
print('data shape')
print(orig_data.shape)



#define testset and trainset indices
nset = round(ndata/(nt+1))

print(nset)
test_ind = np.array([],dtype='int')
for foo in range(nset):
    rand_ind = np.random.permutation(np.arange(foo*(nt+1)+1,(foo+1)*(nt+1)))[:int(0.1*(nt+1))]
    test_ind = np.append(test_ind,rand_ind)
train_ind = np.setdiff1d(np.arange(ndata),test_ind)

# set trainset and testset
trainset = orig_data[train_ind]
testset = orig_data[test_ind] 

# print dataset shapes
print('trainset shape: ', trainset.shape)
print('testset shape: ', testset.shape)

# set dataset
dataset = {'train':data_utils.TensorDataset(torch.tensor(trainset)),
           'test':data_utils.TensorDataset(torch.tensor(testset))}
print(dataset['train'].tensors[0].shape, dataset['test'].tensors[0].shape)

# compute dataset shapes
dataset_shapes = {'train':trainset.shape,
                 'test':testset.shape}

print(dataset_shapes['train'],dataset_shapes['test'])
num_epochs_save_model = 9999999
plt_fname = './training_loss.png'


# set the number of nodes in each layer
input_nodes = [1001, 100, 5]

#set up AE
AE = Autoencoder_pytorch.AE(input_nodes, act_type=1)
encoder = AE.encoder
decoder = AE.decoder

# autoencoder filename
#if not os.path.exists(os.getcwd() + '/model/'):os.makedirs(os.getcwd() + '/model/')
model_fname = './model/1DB.tar'
chkpt_fname = './model/checkpoint.tar'

# set data loaders
train_loader = DataLoader(dataset=dataset['train'],
                        batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=dataset['test'],
                        batch_size=batch_size, shuffle=True, num_workers=2)
data_loaders = {'train':train_loader, 'test':test_loader}

# set device
device = getDevice()

# load model
try:
    checkpoint = torch.load(chkpt_fname, map_location=device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10) 
    
    loss_func = nn.MSELoss(reduction='mean')
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    last_epoch = checkpoint['epoch']
    loss_hist = checkpoint['loss_hist']
    best_loss = checkpoint['best_loss']
    early_stop_counter = checkpoint['early_stop_counter']
    best_encoder_wts = checkpoint['best_encoder_wts']
    best_decoder_wts = checkpoint['best_decoder_wts']
    
    print("\n--------checkpoint restored--------\n")
    
    # compute sparsity in mask
    mask = decoder.state_dict()['full.2.weight_mask']
    print("Sparsity in {} by {} mask: {:.2f}%".format(
        mask.shape[0], mask.shape[1], 100. * float(torch.sum(mask == 0))/ float(mask.nelement())))

    # resume training
    print("")
    print('Re-start {}th training... m={}, f={}, M1={}, M2={}'.format(
        last_epoch+1, encoder.m, encoder.f, encoder.M, decoder.M))
except:
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10) 
    
    loss_func = nn.MSELoss(reduction='mean')
    
    last_epoch = 0
    loss_hist = {'train':[],'test':[]}
    best_loss = float("inf")
    early_stop_counter = 1
    best_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_decoder_wts = copy.deepcopy(decoder.state_dict())
    
    print("\n--------checkpoint not restored--------\n")
pass

# train model
since = time.time()

for epoch in range(last_epoch+1,num_epochs+1):   

    if epoch%num_epochs_print == 0:
        print()
        if scheduler !=None:
            print('Epoch {}/{}, Learning rate {}'.format(
                epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            print('Epoch {}/{}'.format(
                epoch, num_epochs))
        print('-' * 10)

    # Each epoch has a training and test phase
    for phase in ['train', 'test']:
        if phase == 'train':
            encoder.train()  # Set model to training mode
            decoder.train()  # Set model to training mode
        else:
            encoder.eval()   # Set model to evaluation mode
            decoder.eval()   # Set model to evaluation mode
            
        running_loss = 0.0

        AE = AE.to(device)
        # Iterate over data
        for data, in data_loaders[phase]:
            inputs = data.to(device)
            targets = data.to(device)
            

            if phase == 'train':
              
                if scheduler != None:
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    
                    outputs = AE.forward(inputs)#decoder(encoder(inputs))
                   
                    loss = loss_func(outputs, targets)
                   
                    # backward
                    loss.backward()

                    # optimize
                    optimizer.step()  
                
                    # add running loss
                    running_loss += loss.item()*inputs.shape[0]
                    
                else:
                    #print('here')
                    def closure():
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        outputs = decoder(encoder(inputs))
                        loss = loss_func(outputs,targets)

                        # backward
                        loss.backward()
                        return loss

                    # optimize
                    optimizer.step(closure)
                    
                    # add running loss
                    with torch.set_grad_enabled(False):
                        outputs = decoder(encoder(inputs))
                        running_loss += loss_func(outputs,targets).item()*inputs.shape[0]
            else:
                with torch.set_grad_enabled(False):
                    outputs = decoder(encoder(inputs))
                    running_loss += loss_func(outputs,targets).item()*inputs.shape[0]
  
        # compute epoch loss
        epoch_loss = running_loss / dataset_shapes[phase][0]
        loss_hist[phase].append(epoch_loss)
            
        # update learning rate
        if phase == 'train' and scheduler != None:
            scheduler.step(epoch_loss)

        if epoch%num_epochs_print == 0:
            print('{} MSELoss: {}'.format(
                phase, epoch_loss))

    # deep copy the model
    if loss_hist['test'][-1] < best_loss:
        best_loss = loss_hist['test'][-1]
        early_stop_counter = 1
        best_encoder_wts = copy.deepcopy(encoder.state_dict())
        best_decoder_wts = copy.deepcopy(decoder.state_dict())
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:  
            break

    
    # save checkpoint every num_epoch_print
    if epoch%num_epochs_print== 0:
        torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_hist': loss_hist,
                    'best_loss': best_loss,
                    'early_stop_counter': early_stop_counter,
                    'best_encoder_wts': best_encoder_wts,
                    'best_decoder_wts': best_decoder_wts,
                    }, chkpt_fname)

    if epoch%num_epochs_save_model==0:
        print("Saving after {}th training to".format(epoch),
            model_fname )
        torch.save( { 'encoder_state_dict': encoder.state_dict(), 
                    'decoder_state_dict': decoder.state_dict()}, 
                    model_fname )
        # plot train and test loss
        plt.figure()
        plt.semilogy(loss_hist['train'])
        plt.semilogy(loss_hist['test'])
        plt.legend(['train','test'])
        plt.savefig(plt_fname)


print()
print('Epoch {}/{}, Learning rate {}'.format(epoch, num_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
print('-' * 10)
print('train MSELoss: {}'.format(loss_hist['train'][-1]))
print('test MSELoss: {}'.format(loss_hist['test'][-1]))


time_elapsed = time.time() - since

# load best model weights
encoder.load_state_dict(best_encoder_wts)
decoder.load_state_dict(best_decoder_wts)

# compute best train MSELoss
# encoder.to('cpu').eval()
# decoder.to('cpu').eval()

with torch.set_grad_enabled(False):
  train_inputs = torch.tensor(trainset).to(device)
  train_targets = torch.tensor(trainset).to(device)
  train_outputs = decoder(encoder(train_inputs))
  train_loss = loss_func(train_outputs,train_targets).item()

# print out training time and best results
print()
if epoch < num_epochs:
  print('Early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'.format(epoch-last_epoch, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
else:
  print('No early stopping: {}th training complete in {:.0f}h {:.0f}m {:.0f}s'.format(epoch-last_epoch, time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
print('-' * 10)
print('Best train MSELoss: {}'.format(train_loss))
print('Best test MSELoss: {}'.format(best_loss))

###### save models ########
print()
print("Saving after {}th training to".format(epoch),
    model_fname)
torch.save( {'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, 
          model_fname )
# plot train and test loss
plt.figure()
plt.semilogy(loss_hist['train'])
plt.semilogy(loss_hist['test'])
plt.legend(['train','test'])
#plt.show()   
plt.savefig(plt_fname)

# delete checkpoint
try:
  os.remove(chkpt_fname)
  print()
  print("checkpoint removed")
except:
  print("no checkpoint exists")
  
torch.cuda.empty_cache()
