import numpy as np
import re,random

class svm:
    def __init__(self):
        self.dataMat=None
        self.labelVec=None
        self.alpha=None
        self.b=None

    def load(self,fileName=None):
        '''
        This function is used to load train data
        The file should be multiple lines, each line represent a training example
        For each example, the last should be the label, others be the feature, which can be split by a blank
        :param fileName: A string represent the training data
        :return: no value
        '''
        content=""
        data=[]
        label=[]
        with open(fileName,"r",encoding="utf-8") as f:
            content=f.read()
            lines=re.split("\n",content,flags=re.S|re.M)
            for line in lines:
                lineData=list(map(lambda x:float(x),re.split(r" ",line,flags=re.S|re.M)))
                data.append(lineData[0:len(lineData)-1])
                label.append(lineData[-1])
        self.dataMat=np.mat(data)
        self.labelVec=np.mat(label).transpose()

    def __calcFunctionValue(self,i):
        return np.multiply(self.alpha,self.labelVec).T*(self.dataMat*(self.dataMat[i,:].T))+self.b

    def __getRandj(self,i,m):
        ret=i
        while i==ret:
            ret=int(random.uniform(0,m))
        return ret

    def __chipAlpha(self,alpha,L,H):
        if alpha<L:
            return L
        elif alpha>H:
            return H
        else:
            return alpha

    def calc(self,C,tol,maxIter=100):
        '''
        This function is smo algorithm
        Must be called for caculation
        :param C: The pernalty parameter
        :param tol: The error limit
        :param maxIter: the round to train
        :return: self.alpha, self.b
        '''
        m,n=np.shape(self.dataMat)# m train example, n features
        self.alpha=np.mat(np.zeros((m,1)))
        self.b=0.

        for iter in range(0,maxIter):
            #print("pp")
            for i in range(0,m): # for m alpha
                Fxi=self.__calcFunctionValue(i)
                Exi=Fxi-float(self.labelVec[i])
                if(((self.labelVec[i]*Exi<-tol) and (self.alpha[i]<C)) or ((self.labelVec[i]*Exi>tol) and (self.alpha[i]>0))):
                    j=self.__getRandj(i,m)
                    Fxj=self.__calcFunctionValue(j)
                    Exj=Fxj-float(self.labelVec[j])

                    alphaIold=self.alpha[i].copy()
                    alphaJold=self.alpha[j].copy()

                    # calculate L,H
                    if(self.labelVec[i]!=self.labelVec[j]):
                        L=float(max(0,self.alpha[j]-self.alpha[i]))
                        H=float(min(C,C+self.alpha[j]-self.alpha[i]))
                    else:
                        L=float(max(0,self.alpha[j][0]+self.alpha[i][0]-C))
                        H=float(min(C,self.alpha[i]+self.alpha[j]))
                    if L==H:
                        iter-=1
                        continue

                    # calculate \eta
                    eta=self.dataMat[i,:]*self.dataMat[i,:].T+self.dataMat[j,:]*self.dataMat[j,:].T-2.0*self.dataMat[i,:]*self.dataMat[j,:].T

                    # update \alpha_j
                    self.alpha[j]=self.__chipAlpha(self.alpha[j]+self.labelVec[j]*(Exi-Exj)/eta,L,H)

                    # update \alpha_i
                    self.alpha[i]=self.__chipAlpha(self.alpha[i]+self.labelVec[i]*self.labelVec[j]*(alphaJold-self.alpha[j]),L,H)

                    # calculate \b_1 \b_2
                    b1=self.b-Exi-self.labelVec[i]*(self.alpha[i]-alphaIold)*self.dataMat[i,:]*self.dataMat[i,:].T \
                        -self.labelVec[j]*(self.alpha[j]-alphaJold)*self.dataMat[i,:]*self.dataMat[j,:].T
                    b2=self.b-Exj-self.labelVec[i]*(self.alpha[i]-alphaIold)*self.dataMat[i,:]*self.dataMat[j,:].T \
                        -self.labelVec[j]*(self.alpha[j]-alphaJold)*self.dataMat[j,:]*self.dataMat[j,:].T

                    # update b
                    if(self.alpha[i]>0 and self.alpha[i]<C):
                        self.b=b1
                    elif(self.alpha[j]>0 and self.alpha[j]<C):
                        self.b=b2
                    else:
                        self.b=(b1+b2)/2

if __name__=="__main__":
    testSvm=svm()
    testSvm.load('''..//test//test.txt''')
    testSvm.calc(0.6,0.001,40)
    print(testSvm.alpha,testSvm.b)
