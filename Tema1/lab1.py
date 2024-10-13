def read(name):
    A = [[],[],[]]
    B = []
    with open(name, 'r') as file:
        i = 0
        for line in file:
            line = line.replace(" ", "")
            s = 0
            j = line.index('x')
            A[i].append(1 if line[s:j] == '' or line[s:j] == '+' else (-1 if line[s:j] == '-' else int(line[s:j])))
            s = j + 1
            j = line.index('y')
            A[i].append(1 if line[s:j] == '' or line[s:j] == '+' else (-1 if line[s:j] == '-' else int(line[s:j])))
            s = j + 1
            j = line.index('z')
            A[i].append(1 if line[s:j] == '' or line[s:j] == '+' else (-1 if line[s:j] == '-' else int(line[s:j])))
            s = j + 2
            j = len(line)
            B.append(int(line[s:j]))
            i += 1
    return A, B

def det(A):
    return (A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1]) 
    - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0]) 
    + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]))

def trace(A):
    return A[0][0] + A[1][1] + A[2][2]

def vnorm(B):
    return (B[0]**2+B[1]**2+B[2]**2)**0.5

def transpose(A):
    return [[A[0][0],A[1][0],A[2][0]],
    [A[0][1],A[1][1],A[2][1]],
    [A[0][2],A[1][2],A[2][2]]]

def dot(A,B):
    n = len(A)
    return[sum(A[i][j]*B[j] for j in range(n))for i in range(n)]

def cramer(A,B):
    d = det(A)
    r = transpose(A)
    ax = transpose([B,r[1],r[2]])
    ay = transpose([r[0],B,r[2]])
    az = transpose([r[0],r[1],B])
    return [det(ax)/d, det(ay)/d, det(az)/d]

def inversion(A,B):
    n = len(A)
    coef = [[],[],[]]
    for i in range(n):
        for j in range(n):
            nr = []
            for ip in range(n):
                if ip == i:
                    continue
                for jp in range(n):
                    if jp == j:
                        continue
                    nr.append(A[ip][jp])
            coef[i].append((-1)**(i+j)*(nr[0]*nr[3]-nr[1]*nr[2]))
    adj = transpose(coef)
    determ = det(A)
    inv = [[adj[i][j] / determ for j in range(n)] for i in range(n)]
    return dot(inv,B)   

A, B  = read('input1.txt')
print(A)
print(B)
print(det(A))
print(trace(A))
print(vnorm(B))
print(transpose(A))
print(dot(A,B))
print(cramer(A,B))
print(inversion(A,B))


        