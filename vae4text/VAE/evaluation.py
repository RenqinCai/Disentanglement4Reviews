from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import torch
from metric import *

def eval(model, optimizer, loss_func, args):
    batch_size = args.batch_size

    ### data_loader
    data_loader = DataLoader(dataset=datasets['valid'], batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=torch.cuda.is_available())

    ### mode
    model.eval()

    total_loss = 0

    ### data iteration
    for iteration, batch in enumerator(data_loader):

        batch_size = batch['input'].size(0)
        print("batch_size")

        ### Forward pass
        logp, mean, logv, z = model(batch['input'], batch['length'])

        ### loss cal
        NLL_loss, KL_loss, KL_weight = get_loss(loss_func, logp, batch['target'], batch['length'], mean, logv, args.anneal_func, step, args.k, args.x0)

        loss = (NLL_loss+KL_weight*KL_loss)/batch_size

        if iteration == 0:
            total_loss = loss
        else:
            total_loss += loss
        
    return loss
