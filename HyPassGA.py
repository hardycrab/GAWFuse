### coder:xiezhijie
### To make the password dictionary cover more target passwords,
### several password dictionaries are rearranged and combined to generate a new password dictionary.
### model_list = ('Markov','PCFG','RNN')
### source_list = ('000webhost','ClixSense','csdn','linkedin','myspace','youku','yy')
### python guessfuse_GA.py -cf /home/lab/Desktop/password/pcfg_cracker-4.0-rc3/result/000webhost-1000000.txt /home/lab/Desktop/password/new_disk/cross-site/fla/000webhost-1000000.txt /home/lab/Desktop/password/new_disk/cross-site/markov/000webhost-1000000.txt 
#### -tf /home/lab/Desktop/password/new_disk/cross-site/train/000webhost-1000000.txt -sf pass/000webhost.txt -fn 100 -wf weight/000webhost.txt -on 100000
import file_io
from combination import combination
from Genetic_Algorithm import Genetic_Algorithm
import json
import argparse
import os.path
import time

parser = argparse.ArgumentParser(description='calculate the coverage between two *.txt files and show the figure')
parser.add_argument('--target_filename','-ts', required=True)
parser.add_argument('--combined_file','-cf', required=False,nargs='+',default= '')
parser.add_argument('--test_file','-tf', required=False, default= '')
parser.add_argument('--save_file','-sf', required=False, default= '')
parser.add_argument('--first_combined_size','-fn', required=False, default= 10)
parser.add_argument('--weight_file','-wf', required=False, default= '')
parser.add_argument('--output_size','-on', required=False, default= 10000000)
try:
    args=parser.parse_args()
except Exception as msg:
    print(msg)

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'Start Processing ~')
first_combined_size = int(args.first_combined_size)
output_size = int(args.output_size)
target_source = args.target_filename.split('-')[0]
# ,
# print(args.combined_file)
if args.combined_file == ['PFM']:
    combined_file = ['/home/lab/Desktop/password/new_disk/cross-site/pcfg-new/'+target_source+'.txt','/home/lab/Desktop/password/new_disk/cross-site/markov/'+target_source+'-1000000.txt','/home/lab/Desktop/password/new_disk/cross-site/fla/'+target_source+'-1000000.txt']
elif args.combined_file == ['PM']:
    combined_file = ['/home/lab/Desktop/password/new_disk/cross-site/pcfg-new/'+target_source+'.txt','/home/lab/Desktop/password/new_disk/cross-site/markov/'+target_source+'-1000000.txt']
elif args.combined_file == ['PRFM']:
    combined_file = ['/home/lab/Desktop/password/new_disk/cross-site/pcfg-new/'+target_source+'.txt','/home/lab/Desktop/password/new_disk/cross-site/markov/'+target_source+'-1000000.txt','/home/lab/Desktop/password/new_disk/cross-site/fla/'+target_source+'-1000000.txt','/home/lab/Desktop/password/new_disk/cross-site/rfguess-new/'+target_source+'.txt']
else:
    combined_file = args.combined_file

if args.test_file == '':
    test_file = '/home/lab/Desktop/password/new_disk/cross-site/test/'+target_source+'-test.txt'
else:
    test_file = args.test_file

if args.save_file == '':
    save_file = 'pass/'+args.target_filename+'.txt'
else:
    save_file = args.save_file
file_io.filename_exist(save_file)

# save_file = args.save_file
if args.weight_file == '':
    weight_file = 'weight/'+args.target_filename+'.txt'
else:
    weight_file = args.weight_file
# file_io.filename_exist(weight_file)
# model_list = ('pcfg','Markov','LSTM')
# first_combined_size = 3
# #########
# source = '000webhost'
# test_file = '000webhost-train1.txt'
# save_file = 'pass/2time_guessfuse_'+source+'_'+test_file
# weight_file = 'weight/2time_guessfuse_'+source+'_'+test_file
##############

if os.path.isfile(weight_file):
    file=open(weight_file,'r')
    weight_vector_dic = json.loads(file.read())
    file.close()
else:
    weight_vector_dic = {}
# weight_vector_dic = {}
time = 0
start = 0
combined_password_list = {}
# gen_password_list = []
for filename in combined_file:
    password_list = file_io.file_load_password_list(filename,start,first_combined_size)
    combined_password_list[filename]=password_list#PCFG,MARKOV
    # print(index,password_list)
coverage_password_list = file_io.coverage_filter(combined_password_list)
# for pass_list in coverage_password_list:
#     print(pass_list)
# for sublist in coverage_password_list:
#     print(sublist)
####generate random weight_vector
GA = Genetic_Algorithm()
test = file_io.test_password_list(test_file)
if str(start) not in weight_vector_dic:
    weight_vector = GA.start(coverage_password_list,test_file,first_combined_size)
    generated_password_list = combination(coverage_password_list,weight_vector,first_combined_size)
    generated_password_list, new_password_list = generated_password_list[:first_combined_size],generated_password_list[first_combined_size:]
    for filename in combined_file:
        combined_password_list['old-'+filename] = combined_password_list[filename]
    coverate = test.test(generated_password_list)
    print("weight vector:", (weight_vector))
    print('final coverate:', coverate)
    # print("generated password list: ", generated_password_list)
    # print("generated password list:\n", (generated_password_list))
    file_io.save_file(save_file,generated_password_list)
    # gen_password_list.extend(generated_password_list)
    gen_password_set = set([x[0] for x in generated_password_list])
    weight_vector_dic[str(start)] = {"w":weight_vector,'s':first_combined_size}
else:
    weight_vector = weight_vector_dic[str(start)]['w']
    generated_password_list = combination(coverage_password_list,weight_vector,first_combined_size)
    generated_password_list, new_password_list = generated_password_list[:first_combined_size],generated_password_list[first_combined_size:]
    for filename in combined_file:
        combined_password_list['old-'+filename] = combined_password_list[filename]
    coverate = test.test(generated_password_list)
    print("weight vector:", (weight_vector))
    print('final coverate:', coverate)
    # print("generated password list: ", [x[0] for x in generated_password_list])
    # print("generated password list:\n", (generated_password_list))
    file_io.save_file(save_file,generated_password_list)
    # gen_password_list.extend(generated_password_list)
    gen_password_set = set([x[0] for x in generated_password_list])
size = first_combined_size
while(len(gen_password_set)<output_size):
    start = start + size
    if start >= size*10 and size < 1000000:
        size = size*10

    for filename in combined_file:
        password_list = file_io.file_load_password_list(filename, start, size)
        combined_password_list[filename]=password_list
    for filename in combined_password_list: 
        combined_list = []
        for password in combined_password_list[filename]:
            if password[0] not in gen_password_set:
                combined_list.append(password)
        combined_password_list[filename] = combined_list
        # print(index,combined_list)
    coverage_password_list = file_io.coverage_filter(combined_password_list)
    # for pass_list in coverage_password_list:
    #     print(pass_list)
    # index = 0
    # while(index < len(coverage_password_list)):
    #     if coverage_password_list[index] == []:
    #         del coverage_password_list[index]
    #     else:
    #         index += 1
    # for sublist in coverage_password_list:
    #     print(sublist)
    ####generate random weight_vector
    # print('start GA:')
    if str(start) not in weight_vector_dic:
        weight_vector = GA.start(coverage_password_list,test_file,size)
        generated_password_list = combination(coverage_password_list, weight_vector,size)
        generated_password_list, new_password_list = generated_password_list[:size],generated_password_list[size:]
        for filename in combined_file:
            combined_password_list['old-'+filename].extend(combined_password_list[filename])
        # combined_password_list[len(combined_file)].extend(new_password_list)
        coverate = test.test(generated_password_list)
        print("weight vector:", (weight_vector))
        print('final coverate:', coverate)
        # print("generated password list: ", generated_password_list)
        file_io.save_file(save_file,generated_password_list)
        # gen_password_list.extend(generated_password_list)
        gen_password_set.update([x[0] for x in generated_password_list])
        print("total size: ", len(gen_password_set))
        weight_vector_dic[start] = {"w": weight_vector, 's': size}
        file = open(weight_file,'w',encoding='utf-8')
        line = json.dumps(weight_vector_dic)
        file.writelines(line+'\n')
        file.close()
    else:
        weight_vector = weight_vector_dic[str(start)]['w']
        generated_password_list = combination(coverage_password_list, weight_vector,size)
        generated_password_list, new_password_list = generated_password_list[:size],generated_password_list[size:]
        for filename in combined_file:
            combined_password_list['old-'+filename].extend(combined_password_list[filename])
        coverate = test.test(generated_password_list)
        print("weight vector:", (weight_vector))
        print('final coverate:', coverate)
        # print("generated password list: ", generated_password_list)
        file_io.save_file(save_file,generated_password_list)
        # gen_password_list.extend(generated_password_list)
        gen_password_set.update([x[0] for x in generated_password_list])
        print("total size: ", len(gen_password_set))

print('successful!')