with open('history_layer4_0_511.txt', 'r') as f:
    x = f.readlines()
    V = [False]*512
    for line in x:
        if '||||' in line:
            n = ""
            i = 9
            while line[i] != ' ':
                n += line[i]
                i += 1
            if V[int(n)]:
                print(n)
            V[int(n)] = True
    for i, b in enumerate(V):
        if not b:
            print(i)
