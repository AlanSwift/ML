import numpy,re

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
        self.dataMat=numpy.mat(data)
        self.labelVec=numpy.mat(label).transpose()

    def calc(self,maxIter=100):
        '''
        This function is smo algorithm
        Must be called for caculation
        :return:
        '''
        m,n=numpy.shape(self.dataMat)# m train example, n features
        self.alpha=numpy.shape(numpy.zeros(m,1))
        for iter in range(0,maxIter):
            



if __name__=="__main__":
    testSvm=svm()
    testSvm.load('''/Users/Alan/workspace/ml/lib/svm/test/test.txt''')
    testSvm.calc()
    pass