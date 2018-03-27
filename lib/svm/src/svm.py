import numpy as np,re

class svm:
    def __init__(self):
        self.dataMat=None
        self.labelVec=None
        self.alpha=None
        self.b=None

    def load(self,fileName=None):
        content=""
        data=[]
        label=[]
        with open(fileName,"r",encoding="utf-8") as f:
            content=f.read()
            lines=re.split("\n",content,flags=re.S|re.M)
            for line in lines:
                lineData=re.split(r" ",line,flags=re.S|re.M)
                data.append(lineData[0:len(lineData)-1])
                label.append(lineData[-1])
                print(line)
        self.dataMat=np.mat(data)
        self.labelVec=np.mat(label).transpose()

    def __getFunctionValue(self,i):
        return float(np.multiply(self.alpha,self.labelVec).T*self.dataMat*(self.dataMat[i,:].T))

    def calc(self,C,tol,maxIter=100):
        '''
        This function is smo algorithm
        Must be called for caculation
        :return:
        '''
        m,n=np.shape(self.dataMat)# m train example, n features
        self.alpha=np.shape(np.zeros(m,1))
        self.alpha=np.mat(np.zeros((m,1)))
        self.b=0
        for iter in range(0,maxIter):
            for i in range(0,m):
                fxi=self.__getFunctionValue(i)
                Exi=fxi-float(self.labelVec[i])

            pass



if __name__=="__main__":
    testSvm=svm()
    testSvm.load('''/Users/Alan/workspace/ml/lib/svm/test/test.txt''')
    testSvm.calc()
    pass