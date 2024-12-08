##
# @file testReinforce.py
# @author Keren Zhu
# @date 10/31/2019
# @brief The main for test REINFORCE
#

from datetime import datetime
import os
import torch

import reinforce as RF
from env import EnvGraph as Env

import numpy as np
import statistics

import wandb
from wandb.integration.sb3 import WandbCallback

class AbcReturn:
    def __init__(self, returns):
        self.numNodes = float(returns[0])
        self.level = float(returns[1])
    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numNodes < other.numNodes
        else:
            return self.level < other.level
    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numNodes) == int(self.numNodes)

def testReinforce(filename, ben, target):
    #run = wandb.init(
    # project="RLFinal_AIG_Reduction_8step",
    # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # id = "v6_PPO_v3"
    #)
    run = wandb.init(
     project = "RLFinal_AIG_Reduction_20step on " + ben,
     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
     id = "v7_PPO"
    )

    now = datetime.now()
    dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time ", dateTime)

    env = Env(filename)
    env.target = target
    #vApprox = Linear(env.dimState(), env.numActions())
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 1e-4, RF.FcModelGraph)
    #vApprox.load_model("model/vApprox.pth")
    baseline = RF.Baseline(0)
    vbaseline = RF.BaselineVApprox(env.dimState(), env.numActions(), 1e-4, RF.FcModel)
    #vbaseline.load_model("model/vbaseline.pth")
    #vApprox.load_state_dict(torch.load("model/vbaseline.pth"))
    reinforce = RF.Reinforce(env, 1, vApprox, vbaseline)

    lastfive = []

    # mainpla for about 220, apex 400
    # other 300
    for idx in range(400):
        returns = reinforce.episode(phaseTrain=True)
        seqLen = reinforce.lenSeq
        line = "Iter " + str(idx) + ", NumAnd "+ str(returns[0]) + ", Seq Length " + str(seqLen) + "\n"
        
        
        if idx % 100 == 0:
            print("Testing ")
            print("-----------------------------------------------")
            returns = reinforce.episode(phaseTrain=False)
            seqLen = reinforce.lenSeq
            line = "Iter " + str(idx) + ", NumAnd "+ str(returns[0]) + ", Seq Length " + str(seqLen) + "\n"
            print(line)
            print("-----------------------------------------------")

        wandb.log(
            {
            "step": idx,
            "NumAnd": returns[0],
             "avg_score": returns[1]}
        )
        print(line)
        print("-----------------------------------------------")
        print("Action (Policy Value) > ... > || Total Reward, Remain AndGate ||\n")
        
        #reinforce.replay()
    wandb.finish()
    
    # for testing
    #returns = reinforce.episode(phaseTrain=False)
    #seqLen = reinforce.lenSeq
    #line = "Iter " + str(idx + 1) + ", NumAnd "+ str(returns[0]) + ", Level "+ str(returns[1]) + ", Seq Length " + str(seqLen) + "\n"
    print("Testing ")
    print("-----------------------------------------------")
    #lastfive.sort(key=lambda x : x.level)
    #lastfive = sorted(lastfive)
    returns = reinforce.episode(phaseTrain=False)
    seqLen = reinforce.lenSeq
    line = "Iter " + str(idx) + ", NumAnd "+ str(returns[0]) + ", Seq Length " + str(seqLen) + "\n"
    print(line)
    print("-----------------------------------------------")

    """
    avg_and = 0
    avg_level = 0
    allbest_and = 10000
    allbest_level = 10000

    for idx in range(10):
        best_and = 10000
        best_level = 10000
        for testcase in range(10):
            returns = reinforce.episode(phaseTrain=True)
            # store now best
            if best_and > returns[0][0]:
                best_and = returns[0][0]
                best_level = returns[0][1]
            print('.', end='', flush=True)
        print()
        
        # store avg
        avg_and += best_and
        avg_level += best_level

        # store best
        if allbest_and > best_and:
            allbest_and = best_and
            allbest_level = best_level


    print("-----------------------------------------------")
    print("Average 10 Episode")
    print("NumAnd "+ str(avg_and / 10) + ", Level "+ str(avg_level / 10))
    print("Best Over 100 Episode")
    print("NumAnd "+ str(allbest_and) + ", Level "+ str(allbest_level))
    print("-----------------------------------------------")
    
    resultName = "./results/" + ben + ".csv"
    with open(resultName, 'a') as andLog:
        line = "Test 5 actions with combine = 1 \n Average 10 Episode NumAnd "
        line += str(avg_and / 10)
        line += ", Level "
        line += str(avg_level / 10)
        line += "\n"

        line += "Best Over 100 Episode NumAnd "
        line += str(allbest_and)
        line += ", Level "
        line += str(allbest_level)
        line += "\n"
        andLog.write(line)
    #rewards = reinforce.sumRewards
    """

    # save model
    vApprox.save_model("model/vApprox_" + ben + ".pth")
    vbaseline.save_model("model/vbaseline_" + ben + ".pth")

    """
    with open('./results/sum_rewards.csv', 'a') as rewardLog:
        line = ""
        for idx in range(len(rewards)):
            line += str(rewards[idx]) 
            if idx != len(rewards) - 1:
                line += ","
        line += "\n"
        rewardLog.write(line)
    with open ('./results/converge.csv', 'a') as convergeLog:
        line = ""
        returns = reinforce.episode(phaseTrain=False)
        line += str(returns[0])
        line += ","
        line += str(returns[1])
        line += "\n"
        convergeLog.write(line)
    """




if __name__ == "__main__":
    """
    env = Env("./bench/i10.aig")
    vbaseline = RF.BaselineVApprox(4, 3e-3, RF.FcModel)
    for i in range(10000000):
        with open('log', 'a', 0) as outLog:
            line = "iter  "+ str(i) + "\n"
            outLog.write(line)
        vbaseline.update(np.array([2675.0 / 2675, 50.0 / 50, 2675. / 2675, 50.0 / 50]), 422.5518 / 2675)
        vbaseline.update(np.array([2282. / 2675,   47. / 50, 2675. / 2675,   47. / 50]), 29.8503 / 2675)
        vbaseline.update(np.array([2264. / 2675,   45. / 50, 2282. / 2675,   45. / 50]), 11.97 / 2675)
        vbaseline.update(np.array([2255. / 2675,   44. / 50, 2264. / 2675,   44. / 50]), 3 / 2675)
    """
    
    #testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/C1355.blif", "C1355", 386)
    #testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/C6288.blif", "C6288", 1870)
    testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/C5315.blif", "C5315", 1287)
    #testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/dalu.blif", "dalu", 1000)
    testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/k2.blif", "k2", 1035)
    testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/mainpla.blif", "mainpla", 3386)
    testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/apex1.blif", "apex1", 1881)
    testReinforce("/home/rayksm/rlfinal/benchmarks/mcnc/Combinational/blif/bc0.blif", "bc0", 795)

    #testReinforce("/home/rayksm/rlfinal/benchmarks/arithmetic_epfl/sin.blif", "sin", 1023)

    #testReinforce("/home/rayksm/rlfinal/benchmarks/flowtune_BLIF/bflyabc.blif", "bfly_abc")
    #testReinforce("./bench/MCNC/Combinational/blif/prom1.blif", "prom1")
    #testReinforce("./bench/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    #testReinforce("./bench/MCNC/Combinational/blif/k2.blif", "k2")
    #testReinforce("./bench/ISCAS/blif/c5315.blif", "c5315")
    #testReinforce("./bench/ISCAS/blif/c6288.blif", "c6288")
    #testReinforce("./bench/MCNC/Combinational/blif/apex1.blif", "apex1")
    #testReinforce("./bench/MCNC/Combinational/blif/bc0.blif", "bc0")
    #testReinforce("./bench/i10.aig", "i10")
    #testReinforce("./bench/ISCAS/blif/c1355.blif", "c1355")
    #testReinforce("./bench/ISCAS/blif/c7552.blif", "c7552")
