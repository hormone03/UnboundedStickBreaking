import torch
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd  
from utils import parametrizations
from VAEs import StickBreakingVAE

train_valid_test_splits = (45000, 5000, 10000)

from utils import tensor_tr, tensor_te, 
data_size = tensor_tr.size(0)
input_shape =  tensor_tr.size()
train_dataset, test_dataset, input_shape, input_ndims, nc = dataDir(analysis = True, FashionMNIST = True)
seed = 1234
dataloader_kwargs = {}
CUDA = torch.cuda.is_available()
#download_needed = not os.path.exists('./MNIST')
model_path = 'trained_models'
checkpoint_path = None

if CUDA:
    torch.cuda.manual_seed(seed)
    dataloader_kwargs.update({'num_workers': 1, 'pin_memory': True})

# get datasets
#train_dataset = torchvision.datasets.MNIST('.', train=True, transform=transforms.ToTensor(), download=download_needed)
#test_dataset = torchvision.datasets.MNIST('.', train=False, transform=transforms.ToTensor())

# get dimension info
#input_shape = list(train_dataset.data[0].shape)
#input_ndims = np.product(input_shape)

# define data loaders
train_data = train_dataset.data.reshape(-1, 1, *input_shape)/255   # reshaping and scaling bytes to [0,1]
test_data = test_dataset.data.reshape(-1, 1, *input_shape)/255
pruned_train_data = train_data[:train_valid_test_splits[0]]


#model_names = ['GaussianVAE', 'StickBreakingVAE_Gauss_Logit', 'StickBreakingVAE_GDVAE', 'StickBreakingVAE_GDWO', 'StickBreakingVAE_Dirichlet_dist', 'StickBreakingVAE_GEM', 'StickBreakingVAE_Kumaraswamy', 'RawPixels']
model_names = ['StickBreakingVAE_GDVAE', 'StickBreakingVAE_GDWO', 'StickBreakingVAE_Dirichlet_dist', 'StickBreakingVAE_GEM', 'RawPixels']

xy_sets = ['train_data', 'train_labels', 'test_data', 'test_labels']
#k = [3, 5, 10]
#checkpoint_paths = ['trained_models/StickBreakingVAE_GDVAE/best_checkpoint_StickBreakingVAE_GDVAE_Aug_03_2023_18_23',
 #                   'trained_models/StickBreakingVAE_Dirichlet_dist/best_checkpoint_StickBreakingVAE_Dirichlet_dist_Jul_06_2023_17_45',
  #                  'trained_models/StickBreakingVAE_GEM/best_checkpoint_StickBreakingVAE_GEM_Jul_10_2023_17_33']

#                    '/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/KMNIST/trained_models/StickBreakingVAE_Dirichlet_dist/best_checkpoint_StickBreakingVAE_Dirichlet_dist_Sep_12_2023_10_02',
#                    '/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/KMNIST/trained_models/StickBreakingVAE_GDVAE/best_checkpoint_StickBreakingVAE_GDVAE_Sep_12_2023_09_01',
#                    '/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/KMNIST/trained_models/StickBreakingVAE_GDWO/best_checkpoint_StickBreakingVAE_GDWO_Sep_12_2023_09_31']

checkpoint_paths = ['/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/FashionMNIST/trained_models/StickBreakingVAE_GEM/best_checkpoint_StickBreakingVAE_GEM_Sep_11_2023_23_00',
                    '/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/FashionMNIST/trained_models/StickBreakingVAE_Dirichlet_dist/best_checkpoint_StickBreakingVAE_Dirichlet_dist_Sep_11_2023_10_40',
                    '/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/FashionMNIST/trained_models/StickBreakingVAE_GDVAE/best_checkpoint_StickBreakingVAE_GDVAE_Sep_11_2023_09_58',
                    '/home/akinlolu/Desktop/stick_br/Dirichlet-VAE-main/all_Results/FashionMNIST/trained_models/StickBreakingVAE_GDWO/best_checkpoint_StickBreakingVAE_GDWO_Sep_11_2023_10_22']


def tsne_plot(features_dict, model):
    test_y = features_dict[xy_sets[3]].squeeze()
    n_samples = test_y.shape[0]
    test_x = features_dict[xy_sets[2]].reshape(n_samples, -1)
    
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(test_x)
    df = pd.DataFrame()
    df["y"] = test_y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="")
    
    
    plt.legend('')
    
    #plt.tight_layout()
    plt.savefig(f'tsne/{model}.png')




#parametrization = parametrizations['GD'] 
#parametrization = parametrizations['gdwo'] 
#parametrization = parametrizations['Dir'] 
#parametrization = parametrizations['GEM'] 
#parametrization = parametrizations['Kumar']
#parametrization = parametrizations['GLogit']
#parametrization = parametrizations['gaussian']



#Load model state
def load_model_(checkpoint_path):
    parametrization = parametrizations['GD']
    model = StickBreakingVAE(parametrization).cuda() if CUDA else StickBreakingVAE(parametrization)
    model_state_dict = torch.load(checkpoint_path)['model_state_dict'] #model_state_dict is a field in the saved model
    model.load_state_dict(model_state_dict)

    return model

#Load model state
def load_model(checkpoint_path):
    if 'StickBreakingVAE_Gaussian' in checkpoint_path:
        parametrization = parametrizations['gaussian']
    elif 'StickBreakingVAE_Gauss_Logit' in checkpoint_path:
        parametrization = parametrizations['GLogit']
    elif 'StickBreakingVAE_GDVAE' in checkpoint_path:
        parametrization = parametrizations['GD']
    elif 'StickBreakingVAE_GDWO' in checkpoint_path:
        parametrization = parametrizations['gdwo']
    elif 'StickBreakingVAE_Dirichlet_dist' in checkpoint_path:
        parametrization = parametrizations['Dir']
    elif 'StickBreakingVAE_GEM' in checkpoint_path:
        parametrization = parametrizations['GEM']
    elif 'StickBreakingVAE_Kumaraswamy' in checkpoint_path:
        parametrization = parametrizations['Kumar']
    else:
        print("Parameterization does not exist")
    model = StickBreakingVAE(parametrization).cuda() if CUDA else StickBreakingVAE(parametrization)

    model_state_dict = torch.load(checkpoint_path)['model_state_dict'] #model_state_dict is a field in the saved model
    model.load_state_dict(model_state_dict)

    return model

def get_models_dict(train_data, test_data):
    # create nested dict
    models_dict = dict(zip(model_names, [{} for x in model_names]))

    # get data and labels
    train_data = train_data.reshape(-1, 1, *input_shape)[:train_valid_test_splits[0]]
    test_data = test_data.reshape(-1, 1, *input_shape)
    train_labels = train_dataset.targets[:train_valid_test_splits[0]]
    test_labels = test_dataset.targets

    if CUDA:
        train_data = train_data.cuda()
        test_data = test_data.cuda()

    # get raw data features
    features_dict = dict(zip(xy_sets, [train_data.cpu(), train_labels.cpu(),
                                       test_data.cpu(), test_labels.cpu()]))
    models_dict['RawPixels'] = features_dict
    
    # get latent space data features
    for checkpoint_path in checkpoint_paths:
        
        model = load_model(checkpoint_path)
        
        model_name = [x for x in model_names if x in checkpoint_path][0] #checkPoint=[modelLatent, rawPixe] ->why we have index
        
        latent_train_data = model.reparametrize(*model.encode(train_data), parametrization=model.parametrization)
        latent_test_data = model.reparametrize(*model.encode(test_data), parametrization=model.parametrization)
        
        features_dict = dict(zip(xy_sets, [latent_train_data.detach().cpu().numpy(), train_labels.detach().cpu().numpy(),
                                           latent_test_data.detach().cpu().numpy(), test_labels.detach().cpu().numpy()]))
        models_dict[model_name] = features_dict
        
    return models_dict


def main():

        
    models_dict = get_models_dict(train_data, test_data)
    for model in model_names:
        tsne_plot(models_dict[model], model)  

if __name__ == '__main__':
    main()
