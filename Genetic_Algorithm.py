import numpy as np
import file_io
from combination import combination
import time
import multiprocessing
import itertools
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D




# def F(x, y):
#     return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
#         -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)

#
# def plot_3d(ax):
#     X = np.linspace(*X_BOUND, 100)
#     Y = np.linspace(*Y_BOUND, 100)
#     X, Y = np.meshgrid(X, Y)
#     Z = F(X, Y)
#     ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
#     ax.set_zlim(-10, 10)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.pause(3)
#     plt.show()

class Genetic_Algorithm():

    def __init__(self):
        # self.DNA_SIZE = 32 ##DNA长度
        self.POP_SIZE = 50 ##种子数(400)
        # self.mutate_SIZE = 1 ##变异的大小
        self.step_size = 0.1 ##步进数
        self.max_fitness_time = 5##最大值不变次数
        self.CROSSOVER_RATE = 0.8 ##父母交叉率
        self.MUTATION_RATE = 0.3 ##变异率
        self.N_GENERATIONS = 100 ##最大迭代次数
        self.BOUND = [0, 1]


    def start(self,combined_password_list,test_file,combined_size):
        self.number_of_combined_list = len(combined_password_list)
        self.combined_password_list = combined_password_list
        self.test = file_io.test_password_list(test_file)
        self.combined_size = combined_size
        # print(test_file)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), str(0)+' generation',end = ' ')
        ####generate random weight_vector,then caculate the combination
        pop = self.gen_pop()
        #pop = self.gen_all_metrics()
        self.fitness_vector_dic = {}
        # for seed in pop:
        #     self.fitness_vector_dic[str(seed)] = 0.0
        # pop = self.gen_pop()
        # pop = np.array(pop)
        # self.POP_SIZE = len(pop)
        #print(self.POP_SIZE)
        #####################
        flag = 0
        # pop = np.array(self.crossover_and_mutation(pop))
        fitness,weight_vector_array = self.get_fitness(pop)
        # generated_password_list = combination(self.combined_password_list,weight_vector,self.combined_size)
        # max_coverate = self.test.test(generated_password_list[:self.combined_size])
        max_fitness_index = np.argmax(fitness)
        old_fitness = fitness[np.argmax(fitness)]
        output = weight_vector_array[max_fitness_index]
        # print(fitness[np.argmax(fitness)],max_fitness_index,fitness[:5],len(pop),weight_vector_array[max_fitness_index])
        print(fitness[np.argmax(fitness)],len(pop),weight_vector_array[max_fitness_index])
        #print(max_fitness)
        # generated_password_list = combination(self.combined_password_list,output)[:combined_size]
        # coverate = self.test.test(generated_password_list)
        # if coverate != old_fitness:
        #     print('wrong!',np.argmax(fitness))
        #     return 0
        for _ in range(self.N_GENERATIONS-1):  # 迭代N代
            pop = self.select(pop, fitness)  # 选择生成新的种群
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), str(_+1)+' generation',end = ' ')
            pop = np.array(self.crossover_and_mutation(pop))
            fitness,weight_vector_array = self.get_fitness(pop)
            max_fitness = fitness[np.argmax(fitness)]
            max_fitness_index = np.argmax(fitness)
            print(fitness[np.argmax(fitness)],len(pop),weight_vector_array[max_fitness_index])
            # print(max_fitness,max_fitness_index,fitness[:5],len(pop),weight_vector_array[max_fitness_index])
            # print("当代最优权重：", weight_vector_array[np.argmax(fitness)]) 
            if max_fitness == old_fitness:
                flag += 1
                output = weight_vector_array[max_fitness_index]
                if flag >= self.max_fitness_time:
                    print('output final vector')
                    return  output.tolist()
            if max_fitness > old_fitness:
                old_fitness = max_fitness
                # weight_vector = self.translateDNA(pop)
                #print("最优的基因型：", pop[max_fitness_index])     
                output = weight_vector_array[max_fitness_index]
                flag = 0
        
            # else:
            #     flag = 0
        # generated_password_list = combination(self.combined_password_list,output,self.combined_size)
        # coverate = self.test.test(generated_password_list[:self.combined_size])
        # print('final coverate ', coverate)
        # fitness = self.get_fitness(pop)
        # max_fitness_index = np.argmax(fitness)
        # weight_vector = self.translateDNA(pop)
        # #print("最优的基因型：", pop[max_fitness_index])     
        # output = weight_vector[max_fitness_index]
        print('output final vector')
        return  output.tolist()

    #### 生成种群
    def gen_pop(self):
        matrix = np.random.rand(self.POP_SIZE, self.number_of_combined_list)
        # 将矩阵的每一行标准化，使得每行的和为1
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        return matrix
    
    def gen_all_metrics(self):
        # 步进值为0.1，所以可能的值有11个（0, 0.1, ..., 1.0）
        values = [i * self.step_size for i in range(1,int(1/self.step_size) + 1)]
        # 生成所有可能的n元组，每个元组的和为1
        pop = []
        for combo in itertools.product(values, repeat=self.number_of_combined_list):
            if sum(combo) == 1:
                pop.append(combo)
        return np.array(pop)

    ### 适应性函数
    def get_fitness(self,weight_vector_array):
        # weight_vector_array = self.translateDNA(pop)
        # coverate_array, vector_array= [],[]
        fittness_array = []
        pool = multiprocessing.Pool(processes=100)
        # q = multiprocessing.Manager().Queue()
        process_list = []
        for index,weight_vector in enumerate(weight_vector_array):
            if str(weight_vector) in self.fitness_vector_dic:
                coverate = self.fitness_vector_dic[str(weight_vector)]
                result = [coverate,index,weight_vector]
                process_list.append([result,0])
            else:
                res = pool.apply_async(self.get_fitness_child, (weight_vector,index))
                # p.start()
                process_list.append([res,1])
            # print(index)
        pool.close()
        #pool.join()
        # while(q.empty() != True):
        #     [coverate,index,vector_array] = q.get()
        #     fittness_array.append([coverate,index,vector_array])
        for res in process_list:
            if res[1] == 1:
                fittness_array.append(res[0].get())
            else:
                fittness_array.append(res[0])
        pool.join()
        # for index,weight_vector in enumerate(weight_vector_array):
        #     self.get_fitness_child(weight_vector,q,index)
        #     [coverate,index,vector_array] = q.get()
        #     fittness_array.append([coverate,index,vector_array])
        fittness_array.sort(key=lambda x: x[1])
        # print([x[1] for x in fittness_array])
        coverate_array, vector_array= [],[]
        for items in  fittness_array:
            coverate_array.append(items[0])
            vector_array.append(items[2])
        # print(coverate_array)
        coverate_array = np.array(coverate_array)
        # print(coverate_array)
        vector_array = np.array(vector_array)
        return coverate_array, vector_array

    def get_fitness_child(self,weight_vector,index):
        #generated_password_list = combination(self.combined_password_list,weight_vector)[:self.combined_size]
        generated_password_list = combination(self.combined_password_list,weight_vector,self.combined_size)
        fitness = self.test.test(generated_password_list[:self.combined_size])
        # fitness = self.test.test_rbo(generated_password_list[:self.combined_size]) #没用！！！！！
        # q.put([coverate,index,weight_vector])
        return [fitness,index,weight_vector]
        # q.put([coverate,index,weight_vector])


    ### 交叉变异
    def crossover_and_mutation(self,pop):
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            ##归一化
            new_pop.append(father)
            if np.random.rand() < self.CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                child = father.copy()
                mother = pop[np.random.randint(self.POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=1, high=self.number_of_combined_list)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
                if sum(child) != 0:
                    child = child/sum(child)###归一化
                    new_pop.append(child)
            if np.random.rand() < self.MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
                child = father.copy()
                child = self.mutation_multi(child)  # 每个后代有一定的机率发生变异
                if sum(child) != 0:
                    child = child/sum(child)###归一化
                    new_pop.append(child)
            if np.random.rand() < self.MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
                child = father.copy()
                child = self.mutation_cut(child)  # 每个后代有一定的机率发生变异
                if sum(child) != 0:
                    child = child/sum(child)###归一化
                    new_pop.append(child)
            # if np.random.rand() < self.MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            #     child = father.copy()
            #     child = self.mutation_setone(child)  # 每个后代有一定的机率发生变异
            #     if sum(child) != 0:
            #         child = child/sum(child)###归一化
            #         new_pop.append(child)
            if np.random.rand() < self.MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
                child = father.copy()
                child = self.mutation_setzero(child)  # 每个后代有一定的机率发生变异
                if sum(child) != 0:
                    child = child/sum(child)###归一化
                    new_pop.append(child)
        return new_pop

    ###变异
    def mutation_multi(self,child):
        mutate_point = np.random.randint(0, self.number_of_combined_list)  # 随机产生一个实数，代表要变异权重的位置
        child[mutate_point] = child[mutate_point]*10  # 将某个权重乘10
        return child
    
    def mutation_cut(self,child):
        mutate_point = np.random.randint(0, self.number_of_combined_list)  # 随机产生一个实数，代表要变异权重的位置
        child[mutate_point] = child[mutate_point]/10  # 将某个权重除10
        return child
    
    def mutation_setone(self,child):
        mutate_point = np.random.randint(0, self.number_of_combined_list)  # 随机产生一个实数，代表要变异权重的位置
        child[mutate_point] = 1  # 将某个权重置1
        return child
    
    def mutation_setzero(self,child):
        mutate_point = np.random.randint(0, self.number_of_combined_list)  # 随机产生一个实数，代表要变异权重的位置
        child[mutate_point] = 1e-10  # 将某个权重置为最小值
        return child
#################################################################################

########################################################################################

    # def select(self,pop, fitness):  # nature selection wrt pop's fitness
    #     pred = (fitness - np.min(fitness)) + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，
    #     ##通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
    #     idx = np.random.choice(len(pred), size=self.POP_SIZE, replace=False,p=(pred) / (pred.sum()))
    #     return pop[idx]
    
# ########################################################################################    
    def select(self, pop, fitness):
        # # 排名选择：确保最佳个体被复制到下一代
        # best_idx = np.argmax(fitness)
        # selected_population = [pop[best_idx]]

        # 排名选择：根据适应度排名
        ranked_indices = np.argsort(fitness)  # 对适应度进行排序，返回排序后的索引
        # ranked_fitness = fitness[ranked_indices]  # 根据排名获取对应的适应度值

        # selected_population = pop[ranked_indices[:self.POP_SIZE]]
        # 计算排名选择的概率，并确保它们归一化
        selection_probs = ranked_indices/ ranked_indices.sum()

        # # 选择剩余的个体
        # remaining_size = self.POP_SIZE - 1  # 减去已经选择的最佳个体
        remaining_indices = np.random.choice(len(selection_probs), size=self.POP_SIZE, replace=True, p=selection_probs)
        
        # # 将选中的个体添加到选中种群中
        # selected_population.extend(pop[remaining_indices])
        selected_population = pop[remaining_indices]

        return selected_population
    

######################################################

    # def print_info(self,pop):
    #     fitness = self.get_fitness(pop)
    #     max_fitness_index = np.argmax(fitness)
    #     print("max_fitness:", fitness[max_fitness_index])
    #     x, y = self.translateDNA(pop)
    #     print("最优的基因型：", pop[max_fitness_index])
    #     print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))





if __name__ == "__main__":
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    # plot_3d(ax)

    model_list = ('Markov', 'pcfg01', 'LSTM')
    first_combined_size = 100
    #########
    source = '000webhost'
    test_file = 'password/000webhost-train1.txt'
    ##############
    time = 0
    start = 0
    combined_password_list = []
    for model in model_list:
        filename = 'password/' + model + '/' + source + '.txt'
        password_tuple = file_io.file_load_password_list(filename, start, first_combined_size)
        # this is the weight,it can be change by GA
        combined_password_list.append(password_tuple)
    GA = Genetic_Algorithm()
    weight_vector = GA.start(combined_password_list,test_file)


    pass
    # for _ in range(N_GENERATIONS):  # 迭代N代
    #     weight_vector = translateDNA(pop,number_of_combined_list)
    #
    #
    #     # if 'sca' in locals():
    #     #     sca.remove()
    #     # sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
    #     # plt.show();
    #     # plt.pause(0.1)
    #     pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
    #     # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
    #     fitness = get_fitness(pop)
    #     pop = select(pop, fitness)  # 选择生成新的种群

    # print_info(pop)
    # plt.ioff()
    # plot_3d(ax)
