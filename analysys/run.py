import torch
import numpy
from tqdm import tqdm



def collect_samples(network, states, next_states, single_output=True):
        features = []
        diff = []

        for state, next_state in tqdm(zip(states, next_states), total=len(states)):
            with torch.no_grad():
                if single_output:
                    features0 = network(torch.tensor(state, device=network.device).unsqueeze(0))
                    features1 = network(torch.tensor(next_state, device=network.device).unsqueeze(0))
                else:
                    _, features0 = network(torch.tensor(state, device=network.device).unsqueeze(0))
                    _, features1 = network(torch.tensor(next_state, device=network.device).unsqueeze(0))

            features.append(features0.cpu())
            diff.append(features1.cpu() - features0.cpu())

        dist = torch.norm(torch.stack(diff).squeeze(1), p=2, dim=1, keepdim=True)

        return torch.stack(features).squeeze(1).numpy(), dist.numpy()
        
#from model_procgen import *
from model_atari import *

if __name__ == "__main__":

    input_shape = (4, 96, 96)

    path_input  = "atari/venture.npy"
    path_result = "atari/venture_result.npy"
    path_model  = "atari/venture/trained/"
    
 
    model = Model(input_shape)   
    model.load(path_model)

    
    data        = numpy.load(path_input, allow_pickle=True).item()
    states      = data['states']
    next_states = data['next_states']
    
    features, dist = collect_samples(model, states, next_states)

    numpy.save(path_result, {'feature': features, 'dist': dist})