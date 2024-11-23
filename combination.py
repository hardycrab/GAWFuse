### coder:xiezhijie
### To make the password dictionary cover more target passwords,
### several password dictionaries are rearranged and combined to generate a new password dictionary.
### input: combined_password_dic = { ('pw1','pw2',...'pwn'):w1, ('pw1','pw2',....'pwn'):w2.............}
### output: generated_password_tup = ('pw1','pw2',.....'pwn'); new_password_tup = ('pw1','pw2',.....)
### mid_password_list = {'pw1':value,'pw2':value}
### s = 0.9; N = 10^3
import file_io
def prob_combination(combined_password_list = [],weight_vector=[]):
    if len(combined_password_list) == 0:
        print('Input Error')
        return
    mid_password_list = {}
    #index_list = zipf(max_len)
    #index_list = range(1,max_len+1)
    for i,password_list in enumerate(combined_password_list):
        for password_prob in password_list:
            if password_prob[0] not in mid_password_list:
                try:
                    mid_password_list[password_prob[0]] = weight_vector[i] * password_prob[1]
                except TypeError:
                    print(weight_vector[i],end=' ')
                    print(type(weight_vector[i]))
                    print(password_prob[1],end=' ')
                    print(type(password_prob[1]))
            else:
                mid_password_list[password_prob[0]] += weight_vector[i] * password_prob[1]
                # mid_password_list[password_prob[0]][1] += 1
    # print(mid_password_list)
    # for password in mid_password_list.keys():
    #     mid_password_list[password] = mid_password_list[password][0]/mid_password_list[password][1]
        #mid_password_list[password] = mid_password_list[password][0]
    generated_password_list = sorted(mid_password_list.items(),key= lambda x: x[1],reverse=True)
    return generated_password_list

def num_combination(combined_password_list = [],weight_vector=[],combined_size = 0):
    if len(combined_password_list) == 0 or len(weight_vector) == 0 or combined_size ==0 or  len(combined_password_list) != len(weight_vector):
        print('Input Error')
        return
    
    generated_password_list = []
    for i,password_list in enumerate(combined_password_list):
        size = int(combined_size*weight_vector[i])
        if size <= len(password_list):
            generated_password_list.extend(password_list[:size])
        else:
            return []
    return generated_password_list


def combination(combined_password_list,weight_vector,combined_size):
    # return num_combination(combined_password_list,weight_vector,combined_size)
    return prob_combination(combined_password_list,weight_vector)


def zipf(N,s=0.9):
    index_list = []
    sum = 0
    for i in range(N):
        value = 1 / ((i + 1) ** s)
        index_list.append(value)
        sum += value

    for i in range(N):
        index_list[i] = index_list[i]/sum

    return index_list


if __name__ == '__main__':
    test_file = '000webhost-train1.txt'
    combined_password_dic = [
        ('qwerty123', '123456a', 'webhost123'),
        ('12345678', '123456789', '1234567'),
        ('win202123', 'tra2608', 'wodtw123'),
    ]
    #weight_vector = [0.10099423533643695, 0.9516307682770948, 0.23724032862426808]
    weight_vector = [0.9516307682770948, 0.10099423533643695, 0.23724032862426808]
    gen,new = combination(combined_password_dic,weight_vector)
    test = file_io.test_password_list(test_file)
    coverate = test.test(gen)
    print(coverate)







