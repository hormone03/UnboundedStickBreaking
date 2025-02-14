import torch
import torch.utils.data
from torch import optim
#from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import os
from utils import Redundancy, uniqueness, set_Data_coherence, parser, diversity, parametrizations, lookahead, tensor_tr, tensor_te, topic_coherence
from VAEs import StickBreakingVAE
import time
from torch.autograd import Variable
import math
#from gensim.models.coherencemodel import CoherenceModel


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_path = 'trained_models'
time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


parametrization = parametrizations['GD']
#parametrization = parametrizations['gdwo']
#parametrization = parametrizations['Dir']
#parametrization = parametrizations['GEM']
#parametrization = parametrizations['Kumar']
#parametrization = parametrizations['GLogit'] ## NB: this gaussian_STBRK
#parametrization = parametrizations['gaussian']

model = StickBreakingVAE(parametrization).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3) #3
#scheduler_redPlat = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
#scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
#scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20,30, 40,60,80], gamma=0.05)

parametrization_str = parametrization if model._get_name() == "StickBreakingVAE" else ''
model_name = '_'.join(filter(None, [model._get_name(), parametrization_str]))
start_epoch = 1


# init save directories
#tb_writer = SummaryWriter(f'logs/{model_name}')
if not os.path.exists(os.path.join(model_path, model_name)):
    os.mkdir(os.path.join(model_path, model_name))
best_test_epoch = None
best_test_loss = None
best_test_model = None
best_test_optimizer = None
stop_training = None
best_reconstruction_loss = None


rec_loss_train = []
KL_loss_train = []
rec_loss_test = []
KL_loss_test = []



def train_batch(epoch, data, train_loss, train_recon_loss, train_KL_loss):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    #mu = mu.long().to(device)
    #logvar = logvar.long().to(device)
    #logistic_z = gauss_z.cpu().detach()
    #Rg_model = Rg_model.fit(data.view(-1, 784).cpu(), Y.cpu())
    #RegAccuracy = Rg_model.score(recon_batch.view(-1, 784).detach().cpu(), Y.cpu())
    batch_recon_loss, batch_KL_loss = model.ELBO_loss(recon_batch, data, mu, logvar, model.kl_divergence,  model.parametrization)
    batch_loss = batch_recon_loss + batch_KL_loss
    batch_loss = batch_loss.mean()
    batch_loss.backward()
    #train_loss += loss.item()
    #train_recon_loss += reconLoss.mean().item()
    #train_KL_loss += analytical_kld.mean().item()
    optimizer.step()
    #scheduler_1.step()
    #scheduler_2.step()
    #if batch_indices % args.log_interval == 0:
        #print(f"gause_z:{gauss_z[0]}") # Variables following a normal distribution after Laplace approximation
        #print(f"dir_z:{dir_z[0]},SUM:{torch.sum(dir_z[0])}") # Variables that follow a Dirichlet distribution. This is obtained by entering gauss_z into the softmax function
        #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
           # epoch, batch_indices * len(data), len(tensor_tr),
            #100. * batch_indices / len(tensor_tr),
            #batch_loss.item() / len(data)))
        #print('Logistic model accuracy = {:.4f}'.format(RegAccuracy))

    #print('====> Epoch: {} Average loss: {:.4f}'.format(
         # epoch, train_loss / len(train_loader.dataset)))
    #print('====> Average logistic model accuracy = {}'.format(RegAccuracy))

    return recon_batch, batch_loss.item(), batch_recon_loss.mean().item(), batch_KL_loss.mean().item()




def test(epoch, data, test_loss, test_recon_loss, test_KL_loss):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        recon_batch_test, mu, logvar = model(data)
        reconLoss, analytical_kld = model.ELBO_loss(recon_batch_test, data, mu, logvar, model.kl_divergence,  model.parametrization)
        batch_loss = reconLoss + analytical_kld
        loss_sum = batch_loss.sum()
        batch_loss = batch_loss.mean()
        #test_loss += loss.mean().item()
        ##########################################
        #scheduler_redPlat.step(test_loss)
        #test_recon_loss += reconLoss.mean().item()
        #test_KL_loss += analytical_kld.mean().item()

    return recon_batch_test, loss_sum.item(), batch_loss.item(), reconLoss.mean().item(), analytical_kld.mean().item()

inference_time = 0
for epoch in range(1, args.epochs + 1):
    train_loss = 0
    train_recon_loss = 0
    train_KL_loss = 0
    data_size = tensor_tr.size(0)
    all_indices = torch.randperm(data_size).split(args.batch_size)
    print_interval = 0
    for batch_indices in all_indices:
        print_interval += 1
        batch_input = Variable(tensor_tr[batch_indices]).to(device)
        rrecon_batch, batch_loss, batch_recon_loss, batch_KL_loss = train_batch(epoch, batch_input, train_loss, train_recon_loss, train_KL_loss)
        train_loss += batch_loss
        train_recon_loss += batch_recon_loss
        train_KL_loss += batch_KL_loss
        if print_interval % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, len(batch_indices) * print_interval, len(tensor_tr),
                100. * len(batch_indices) * print_interval / len(tensor_tr),
                batch_loss / len(batch_indices)))

    train_loss = train_loss / data_size
    train_recon_loss /= data_size
    train_KL_loss /= data_size
    rec_loss_train.append(train_recon_loss)
    KL_loss_train.append(train_KL_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
      epoch, train_loss))


    ############################# TESTING #################################
    test_loss = 0
    test_recon_loss = 0
    test_KL_loss = 0
    word_count = 0
    total_loss = 0

    test_size = tensor_te.size(0)
    test_indices = torch.randperm(test_size).split(args.batch_size)
    start_time = time.time()
    for sample_indices in test_indices:
        test_input = Variable(tensor_te[sample_indices]).to(device)
        word_count += torch.sum(test_input) #add the elements of 2D tensor (freq. of words)
        recon_batch_test, loss_sum, batch_test_loss, test_recon_loss, test_KL_loss = test(epoch, test_input, test_loss, test_recon_loss, test_KL_loss)
        test_loss += batch_test_loss
        total_loss += loss_sum
        test_recon_loss += test_recon_loss
        test_KL_loss += test_KL_loss

    inference_time = time.time() - start_time

    test_loss /= test_size
    test_recon_loss /= test_size
    test_KL_loss /= test_size
    rec_loss_test.append(test_recon_loss)
    KL_loss_test.append(test_KL_loss)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    if  epoch == start_epoch:
        best_test_epoch = epoch
        best_test_loss = test_loss
        best_reconstruction_loss = test_recon_loss
    else:
        best_test_epoch = epoch if test_loss < best_test_loss else best_test_epoch
        best_test_loss = test_loss if best_test_epoch else best_test_loss
        stop_training = True if epoch - best_test_epoch > lookahead else False
        best_reconstruction_loss = test_recon_loss if best_test_epoch else best_reconstruction_loss


    ################### UPDATE Saved_MODEL PARAMETERS If THERE'S IMPROVEMENT ##################
    if epoch == best_test_epoch:
        print("Updating model weights from best epoch")
        best_test_model = model.state_dict().copy()
        best_test_optimizer = optimizer.state_dict().copy()
        #test_loss = 0
        #test_recon_loss = 0
        #test_KL_loss = 0
        #start_time = time.time()
        #beta = model.decoder.weight.detach().cpu().numpy().T
        beta = model.fc.weight.cpu().detach().numpy().T
        #beta = beta.argsort(axis=1)[:, ::-1]
        #dev_input =Variable(tensor_te[sample_indices]).to(device)
        #recon_batch_test, dev_loss, test_recon_loss, test_KL_loss = test(epoch, dev_input, test_loss, test_recon_loss, test_KL_loss)
        #counts = dev_input.sum(1)
        avg = (total_loss / word_count)
        bestPerplexity = math.exp(avg)
        print('===>>> The approximated perplexity is: ', bestPerplexity)

    elif stop_training:
        print("===> Training stopped because there was no further improvement")
        break


print('======>>>> FINAL MODLES Evaluation')
print('===>>> The inference time is : {}'.format(inference_time/test_size))
print('Best epoch is ' + str (best_test_epoch))
print('Best reconstruction error is ' + str(best_reconstruction_loss))
diversity_ = diversity(beta)
print("===>>> The diversity is: ", diversity_)
print('****' * 5)

redundacy_ = Redundancy(beta)
print("===>>> The redundacy is: ", redundacy_)

print('****' * 5)
print('The approximated perplexity is: ', bestPerplexity)

print('*****' * 5)
uniqueness_ = uniqueness(beta)
print("===>>> The uniqueness is: ", uniqueness_)

print('*****' * 5)

data_ = set_Data_coherence(args.dataset_te)
print("Calculating coherence, might take long time")
h, topicCoh = topic_coherence(data_, beta, n_top_words=10)
print("==> Topic  coherence: ", topicCoh)


np.save('outResults/rec_loss_train_' + str(model_name), np.array(rec_loss_train))
np.save('outResults/KL_loss_train_' + str(model_name), np.array(KL_loss_train))
np.save('outResults/rec_loss_test_' + str(model_name), np.array(rec_loss_test))
np.save('outResults/KL_loss_test_' + str(model_name), np.array(KL_loss_test))
#print(rec_loss_test)
#print('#' * 30)
#print(KL_loss_test)
torch.save({'epoch': best_test_epoch,
            'model_state_dict': best_test_model,
            'optimizer_state_dict': best_test_optimizer},
           os.path.join(model_path, model_name, f'best_checkpoint_{model_name}_{time_now}'))