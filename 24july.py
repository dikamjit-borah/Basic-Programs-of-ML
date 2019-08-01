import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#########################################
#definations to be used later
I_nodes=13
H1_nodes=8
H2_nodes=5
O_nodes=3
lr=0.01
#######################################

db=pd.read_csv('W1data.csv')
df=pd.DataFrame(db)
w1,w2,w3=[],[],[]
#w=[w1,w2,w3]
h1,h2,out,h3=[],[],[],[]
b2,b3,b4=[],[],[]
#h3=[]
#out1=[]
#res=[]
#inp=[]
w1,w2,w3=np.random.randn(H1_nodes,I_nodes),np.random.randn(H2_nodes,H1_nodes),np.random.randn(O_nodes,H2_nodes)
b2,b3,b4=np.random.randn(H1_nodes,1),np.random.randn(H2_nodes,1),np.random.randn(O_nodes,1)
cols,inp,res=[],[],[]


def m():
    global cols,inp,res
    for i in range(5000):
        r=np.random.randint(0,173)
        cols=df.iloc[r][0:13]
        inp=cols.values.reshape(13,1)
        res=df.iloc[r][13:16].values.reshape(3,1)
        backprop()
    
    for i in range(5):
        cols=df.iloc[173+i][0:13]
        inp=cols.values.reshape(13,1)
        output=feedforward()
        print("Predicted value")
        print(output)
        print("Actual Value")
        print(df.iloc[173+i][13:16].values.reshape(3,1)) 
m()

def sig(z):
    return 1/(1+np.exp(-z))

def sig_prime(z):
    return z*(1-z)

def cost_derivative(h3):
    return res-h3

def feedforward():
    
    global w1,w2,w3,b2,b3,b4,h1,h2,h3,out;
   
    a1=np.dot(w1,inp)+b2
    h1=sig(a1)    
  
    a2=np.dot(w2,h1)+b3
    h2=sig(a2)
    
    out=np.dot(w3,h2)+b4
    h3=sig(out)
    return h3
    
    
#s1,s2,h33=feedforward()
#h33=feedforw
    
def backprop():
    global w1,w2,w3,b2,b3,b4,h1,h2,h3,out;
    
    delta_w1,delta_b2=[],[]
    delta_w2,delta_b3=[],[]
    delta_w3,delta_b4=[],[]
    
    feedforward()
    error_o=res-h3
    w3_t=np.transpose(w3)
    error_h2=np.dot(w3_t,error_o)
    w2_t=np.transpose(w2)
    error_h1=np.dot(w2_t,error_h2)
    
    
    #delta_w1=np.dot(inp.reshape(13,1),error_h1.reshape(1,8))
    #delta_w2=np.dot(np.asarray(h1).reshape(8,1),error_h2.reshape(1,5))
    
    G_w3=np.multiply(sig_prime(h3),error_o)
    G_w3=np.multiply(lr,G_w3)
    delta_w3=np.dot(G_w3,h2.reshape(1,5))
    
    G_w2=np.multiply(sig_prime(h2),error_h2)
    G_w2=np.multiply(lr,G_w2)
    delta_w2=np.dot(G_w2,h1.reshape(1,8))

    G_w1=np.multiply(sig_prime(h1),error_h1)
    G_w1=np.multiply(lr,G_w1)
    delta_w1=np.dot(G_w1,inp.reshape(1,13))
    
    w1=w1+delta_w1
    w2=w2+delta_w2
    w3=w3+delta_w3

    delta_b2=lr*error_h1
    delta_b3=lr*error_h2
    delta_b4=lr*error_o
    
    b2=b2+delta_b2
    b3=b3+delta_b3
    b4=b4+delta_b4
    
    

               
