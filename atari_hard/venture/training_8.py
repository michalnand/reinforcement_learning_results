import RLAgents

import ppo_cnd_8.src.model_ppo           as ModelPPO
import ppo_cnd_8.src.model_cnd_target    as ModelTarget
import ppo_cnd_8.src.model_cnd           as ModelPredictor
import ppo_cnd_8.src.config              as Config
 
#torch.cuda.set_device("cuda:0")
  
path = "./ppo_cnd_8/"
 
config  = Config.Config() 

#config.envs_count = 1 
 
envs = RLAgents.MultiEnvParallelOptimised("VentureNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count)
#envs = RLAgents.MultiEnvSeq("VentureNoFrameskip-v4", RLAgents.WrapperMontezuma, config.envs_count, True)
#envs = RLAgents.MultiEnvSeq("VentureNoFrameskip-v4", RLAgents.WrapperMontezumaVideo, config.envs_count)
 
agent = RLAgents.AgentPPOCND(envs, ModelPPO, ModelTarget, ModelPredictor, config)

max_iterations = 1000000 
  

trainig = RLAgents.TrainingIterations(envs, agent, max_iterations, path, 128)
trainig.run() 

'''
agent.load(path)
agent.disable_training()

while True:
    reward, done, info = agent.main()
    if done:
        break
'''