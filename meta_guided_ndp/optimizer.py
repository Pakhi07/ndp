from utils import x0_sampling

import cma
import time
import psutil
import numpy as np
from multiprocessing import Pool


def CMAES(config, fitness):
    num_parameters = config["num_parameters"]
    x0 = x0_sampling(config["x0_dist"], num_parameters)
    es = cma.CMAEvolutionStrategy(
        x0,
        config["sigma_init"],
        {
            "verb_disp": config["print_every"],
            "popsize": config["popsize"],
            "maxiter": config["generations"],
            "seed": config["seed"],
            "CMA_elitist": config["CMA_elitist"],
            "minstd": config["minstd"],
        },
    )

    start_time = time.time()

    objective_solution_best = np.Inf
    objective_solution_centroid = np.Inf
    objectives_centroid = []
    objectives_best = []
    gen = 0
    early_stopping_executed = False

    num_cores = (
        psutil.cpu_count(logical=False)
        if config["threads"] == -1
        else config["threads"]
    )

    while not es.stop() or gen < config["generations"]:
        try:
            X = es.ask()
            if num_cores > 1:
                with Pool(num_cores) as pool:
                    fitness_values = pool.map_async(fitness, X).get()

            else:
                fitness_values = [fitness(x) for x in X]

            if config["maximize"]:
                fitness_values = [-f for f in fitness_values]

            es.tell(X, fitness_values)

            if gen % config["print_every"] == 0:
                es.disp()

            objective_current_best_sol = es.best.f
            objectives_best.append(objective_current_best_sol)
            if objective_current_best_sol < objective_solution_best:
                objective_solution_best = objective_current_best_sol
                best_solution = es.best.x

            objetive_current_centroid_sol = es.fit.fit.mean()
            objectives_centroid.append(objetive_current_centroid_sol)
            if objetive_current_centroid_sol < objective_solution_centroid:
                objective_solution_centroid = objetive_current_centroid_sol
                centroid_solution = es.mean

            gen += 1

            if gen % config["evolution_feval_check_every"] == 0:
                test_fevals = []
                for _ in range(
                    config["evolution_feval_check_N"] // config["num_episode_evals"]
                ):
                    checksum = True if _ == 0 else False
                    feval = fitness(best_solution)
                    test_fevals.append(feval)

        except KeyboardInterrupt:
            time.sleep(5)
            break
