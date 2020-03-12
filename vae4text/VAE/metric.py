
def get_loss(NLL, logp, target, length, mean, logv, anneal_func, step, k, x0):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)

    logp = logp.view(-1, logp.size(2))

    NLL_loss = NLL(logp, target)

    KL_loss = -0.5*torch.sum(1+logv-mean.pow(2)-logv.exp())
    KL_weight = kl_anneal_func(anneal_func, step, k, x0)

    return NLL_loss, KL_loss, KL_weight

def kl_anneal_func(anneal_func, step, k, x0):
    if anneal_func == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_func == "linear":
        return min(1, step/x0)

