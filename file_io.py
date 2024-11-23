from typing import Counter
import os.path
import time
# import rbo


def file_load_password_list(filename='', start=0,size=300):
    password_list = []
    file = open(filename,'r',encoding='utf-8')
    i = 0
    while(i < start):
        file.readline()
        i += 1

    # total_prob = 0.0

    for i in range(size):
        line = file.readline()
        if len(line) == 0:
            break
        password = line.rsplit('\t',1)
        password[1] = float(password[1])
        # total_prob += password[1]
        # password = line.strip()
        password_list.append(password)
    
    # for i in range(size):
    #     password_list[i][1] = password_list[i][1]/total_prob

    if len(password_list)>0:
        return password_list
    else:
        print('Load '+filename+'error')
        return

def save_file(filename='test.txt', password_list=[]):
    if len(password_list) == 0:
        print("save error!")
    file = open(filename,'a',encoding='utf-8',newline='')
    for password in password_list:
        file.write(password[0]+'\t'+str(password[1])+'\n')
    file.close()

def filename_exist(filename='test.txt'):
    if os.path.isfile(filename):
            temp = filename[:-4]
            os.rename(filename,temp+str(time.strftime('-%m%d-%H%M',time.localtime(time.time())))+'.txt')


# 定义生成集合交集逻辑标识符的函数
def generate_logics(n_sets):
    """Generate intersection identifiers in binary (0010 etc)"""
    for i in range(1, 2**n_sets):
        yield bin(i)[2:].zfill(n_sets)

# 定义生成Venn图标签和数据的函数
def coverage_filter(combined_password_list):
    """Generate petal descriptions for venn diagram based on set sizes"""
    n_sets = len(combined_password_list)
    label_dic = {}
    for label in generate_logics(n_sets):
        for index,x in enumerate(label):
            if x == '1':
                label_dic[label+str(index)] = {}
    label_list = sorted(label_dic.keys())
    # print(label_list)
    all_pass = {}
    for index,keyname in enumerate(combined_password_list):
        for pass_prob in combined_password_list[keyname]:
            password = pass_prob[0]
            prob = pass_prob[1]
            if password not in all_pass:
                all_pass[password] = {'prob_map':[0.0]*n_sets,'label':['0']*n_sets}
                all_pass[password]['label'][index] = '1'
                all_pass[password]['prob_map'][index] = prob
            else:
                all_pass[password]['label'][index] = '1'
                all_pass[password]['prob_map'][index] = prob
    
    for password in all_pass:
        for index,x in enumerate(all_pass[password]['label']):
            if x == '1':
                prob = all_pass[password]['prob_map'][index]
                label_dic[''.join(all_pass[password]['label'])+str(index)][password] = prob

    for label in label_list:
        label_dic[label] = sorted(label_dic[label].items(),key=lambda x:x[1],reverse=True)

    return list(label_dic.values())



# def coverage_filter(combined_password_list):
#     # input_dic_num = len(combined_password_list)
#     # total_dic_num = 0
#     # for i in range(input_dic_num):
#     #     total_dic_num += math.comb(input_dic_num,i+1)
#     # total_dic,input_dic = [],[]
#     # base_list = combined_password_list.pop()
#     # while(len(combined_password_list)!= 0):
#     #     cover_list = combined_password_list.pop()
#     total_pass = {}
#     for index,input_dic in enumerate(combined_password_list):
#         for pass_prob in input_dic:
#             if pass_prob[0] not in total_pass:
#                 total_pass[pass_prob[0]] = {index:pass_prob[1],}
#             else:
#                 total_pass[pass_prob[0]][index] = pass_prob[1]

#     total_dic = {}
#     #print(len(total_pass))
#     for password in total_pass:
#         if len(total_pass[password]) > 1:
#             tail = ''
#             for key in total_pass[password].keys():
#                 tail += key
#             for item in total_pass[password].items():
#                 source = item[0] + tail
#                 prob = item[1]
#                 if source not in total_dic.keys():
#                     total_dic[source] = [[password,prob],]
#                 else:
#                     total_dic[source].append([password,prob])
#         else:
#             for item in total_pass[password].items():
#                 source = item[0]
#                 prob = item[1]
#                 if source not in total_dic.keys():
#                     total_dic[source] = [[password,prob],]
#                 else:
#                     total_dic[source].append([password,prob])
     
#     # print(total_dic)
#     sorted_total_dic = sorted(total_dic.items(),key=lambda x: x[0])
#     #print(sorted_total_dic)
#     total_list = [pass_list[1] for pass_list in sorted_total_dic]
#     return total_list

def num_coverage_filter(combined_password_list):
    # input_dic_num = len(combined_password_list)
    # total_dic_num = 0
    # for i in range(input_dic_num):
    #     total_dic_num += math.comb(input_dic_num,i+1)
    # total_dic,input_dic = [],[]
    # base_list = combined_password_list.pop()
    # while(len(combined_password_list)!= 0):
    #     cover_list = combined_password_list.pop()
    total_pass = {}
    for index,input_dic in enumerate(combined_password_list):
        for password in input_dic:
            if password[0] not in total_pass:
                total_pass[password] = {str(index):1,}
            else:
                total_pass[password][str(index)] = 1
    total_dic = {}
    #print(len(total_pass))
    for password in total_pass.keys():
        if len(total_pass[password]) > 1:
            tail = ''
            for key in total_pass[password].keys():
                tail += key
            # for item in total_pass[password].items():
            #     source = item[0] + tail
            #     prob = item[1]
            if tail not in total_dic.keys():
                total_dic[tail] = [password,]
            else:
                total_dic[tail].append(password)
        else:
            for item in total_pass[password].items():
                source = item[0]
                #prob = item[1]
                if source not in total_dic.keys():
                    total_dic[source] = [password,]
                else:
                    total_dic[source].append(password)
     
    # print(total_dic)
    sorted_total_dic = sorted(total_dic.items(),key=lambda x: x[0])
    #print(sorted_total_dic)
    total_list = [pass_list[1] for pass_list in sorted_total_dic]
    return total_list



def ratefuse(combined_password_list,test_file,first_combined_size):
    test = test_password_list(test_file)
    rate_list = []
    total_rate = 0.0
    for pass_list in combined_password_list:
        rate = test.test(pass_list[:first_combined_size])
        rate_list.append(rate)
        total_rate += rate
    weight_vector = []
    for rate in rate_list:
        weight_vector.append(rate/total_rate)
    return weight_vector

class test_password_list:
    def __init__(self,test_file):
        self.test_password = []
        file = open(test_file, 'r')
        self.test_password = [x.strip() for x in file.readlines()]
        file.close()
        self.test_password_counter = Counter(self.test_password)
        self.test_number = len(self.test_password)
        self.test_password_list = [x[0] for x in self.test_password_counter.most_common()]
        # self.test_password = set(self.test_password)
        if self.test_number == 0:
            print('Tested Error')
            return IOError

    #test_generated_password_list
    def test(self,generated_password_list):
        # temp = []
        # for line in generated_password_list:
        #     temp.append(line[0])
        # temp = set(temp) & self.test_password
        succeed_number = 0
        for line in generated_password_list:
            if line[0] in self.test_password_counter:
                succeed_number += self.test_password_counter[line[0]]
        ###print(float(succeed_number / self.test_number))
        return float(succeed_number / self.test_number)
    
    def test_rbo(self,generated_password_list):
        generated_password_list = [x[0] for x in generated_password_list]
        test_set = set(generated_password_list)
        for index,password in enumerate(self.test_password_list):
            if password in test_set:
                break
        sim_value = rbo.RankingSimilarity(self.test_password_list[index:], generated_password_list).rbo()
        return sim_value


if __name__ == '__main__':
    filename = '../Markov/ClixSense.txt'
    test_password_list = file_load_password_list(filename,0,size = 6)
    print(test_password_list)
    

