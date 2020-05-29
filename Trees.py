import math
import time
import random
import pandas as pd

class Tree:
    def __init__(self, typ):
        self.type=typ
        if typ==1:
            # classification
            self.months = [1,2,3,4,5,6,7,8,9,10,11,12]
            self.target = 'month'
        elif typ==2:
            self.target = 'pm25'
        self.regions = {}
        self.splits = []
        self.region_res = {}
        self.train_itrs = 3
        self.no_samples = 100

    def clear(self):
        self.regions.clear()
        self.splits.clear()
        self.region_res.clear()
    
    def train(self, data, rf=False):
        print('Training started')
        self.clear()
        self.regions['0']=data
        predictors = []
        for val in data:
            if val!=self.target:
                predictors.append(val)
        print('Predictors:', predictors)
        
        itr = 0
        curgen = []
        while itr<self.train_itrs:
            print('Train Split:', itr)
            curgen.clear()
            # precomputing regional errors
            if rf==True:
                predictors = random.sample(predictors, 5)
            for predictor in predictors:
                print('.', end='')
                curgen.append(self.make_split(predictor))
            curgen.sort()
            print('Curgen:', curgen)
            
            if curgen[0]==(1000000000,'-1',-1,'NAN'):
                return
            # split regions based on results
            marker = curgen[0][1]
            reg = self.regions[marker]
            del self.regions[curgen[0][1]]
            r1 = reg.loc[(reg[curgen[0][3]] <= curgen[0][2])]
            r2 = reg.loc[(reg[curgen[0][3]] > curgen[0][2])]
            self.splits.append((marker,curgen[0][2],curgen[0][3]))
            if itr==0:
                marker=''
            if not r1.empty:
                self.regions[marker+str(itr)+'l']=r1
            if not r2.empty:
                self.regions[marker+str(itr)+'r']=r2
            itr+=1
        if self.type == 1:
            acc = 0
            # saving predictions by region
            for region in self.regions:
                region_counts = self.regions[region][self.target].value_counts()
                self.region_res[region] = region_counts.index[0]
            # computing training accuracy
            for region in self.regions:                
                acc += self.compute_accuracy(self.regions[region], region)
            acc/=len(self.regions)
            print('---------\nTrain Data Classification Accuracy:', acc*100,'\n-----------')
        elif self.type == 2:
            amse = 0
            for region in self.regions:
                pm25s = self.regions[region][self.target]
                mean = pm25s.sum()/self.regions[region].shape[0]
                amse+=mean
                self.region_res[region] = mean
            amse/=len(self.regions)
            print('---------\nTrain Data Regression MSE:', amse,'\n-----------')
            
    def make_split(self, predictor):
        splits = []
        for i, rid in enumerate(self.regions):
            rlows=[]
            j=0
            n=self.no_samples
            #n=len(self.regions[rid][predictor])
            if n>len(self.regions[rid][predictor]):
                n=len(self.regions[rid][predictor])

            while j<=n:
                value = self.regions[rid].sample()[predictor].iloc[0]
                r1 = self.regions[rid].loc[(self.regions[rid][predictor] <= value)]
                r2 = self.regions[rid].loc[(self.regions[rid][predictor] > value)]
                if r1.empty or r2.empty:
                    j+=1
                    continue
                r_err = self.compute_error([r1, r2])
                rlows.append((r_err, rid, value, predictor))
                j+=1
            if rlows==[]:
                return (1000000000,'-1',-1,'NAN')
            rlows.sort()
            splits.append(rlows[0])
        splits.sort()
        return splits[0]
    
    def compute_error(self, regions):        
        total = 0
        for reg in regions:
            total+=reg.shape[0]
        if self.type == 1:
            entropy = 0
            for reg in regions:
                ent = 0
                region_counts = reg[self.target].value_counts()
                rtotal = reg.shape[0]
                for val in region_counts:
                    p = (val/rtotal)
                    ent+= -math.log(p,12)
                    #ent+= p**2
                    #ent+=p*(1-p)
                entropy+=(rtotal/total)*(ent)
            return entropy
        elif self.type == 2:
            err=0
            for reg in regions:
                mse = 0
                pm25s = reg[self.target]
                rtotal = reg.shape[0]
                mean = pm25s.sum()/rtotal
                for pm25 in pm25s:
                    mse+=(mean-pm25)**2
                err+=(rtotal/total)*mse
            return err/len(regions)
            
    def compute_accuracy(self, region, rid):
        pred = 0
        if rid in self.region_res:
            pred = self.region_res[rid]
        else:
            print('Exact region missing from test, generalizing')
            while rid not in self.region_res and rid!='':
                rid = rid[:-2]
            if rid=='':
                if self.type == 1:
                    pred=random.randrange(1, 13, 1)
                else:
                    for reg in self.region_res:
                        pred=self.region_res[reg]
            else:
                pred = self.region_res[rid]

        if self.type == 1:
            region_counts = region[self.target].value_counts()
            acc = 0
            for count, month in zip(region_counts, region_counts.index):
                if month==pred:
                    acc+=count
            acc /= region.shape[0]
            return acc
        elif self.type == 2:
            mse = 0
            pm25s = region[self.target]
            mean = pm25s.sum()/region.shape[0]
            for pm25 in pm25s:
                mse+=(mean-pm25)**2
            return mse/region.shape[0]
        
    def predict(self, data):
        regions = dict()
        regions['0'] = data
        itr = 0
        for split in self.splits:
            rid, targetValue, predictor = split
            if rid in regions:
                targetRegion = regions[rid]
            else:
                continue
            r1 = targetRegion.loc[(targetRegion[predictor] <= targetValue)]
            r2 = targetRegion.loc[(targetRegion[predictor] > targetValue)]        
            del regions[rid]
            if itr!=0:
                rid=rid+str(itr)
            if not r1.empty:
                regions[rid+'l']=r1
            else:
                print(rid+'l IS EMPTY')
            if not r2.empty:
                regions[rid+'r']=r2
            else:
                print(rid+'r IS EMPTY')
            itr+=1
        acc = 0
        print('TEST DATA REGIONS')
        for reg in regions:
            print(reg)
        for reg in regions:
            acc += self.compute_accuracy(regions[reg], reg)
        acc/=len(regions)
        if self.type == 1:
            print('Test Data Classification Accuracy:', acc*100)
            return acc*100
        elif self.type == 2:
            print('Test Data Regression MSE:', acc) 
            return acc
    
    def predictPoint(self, point):
        point_reg = '0'
        #print(point)
        #print(self.region_res)
        #print(self.splits)
        for i, split in enumerate(self.splits):
            rid, targetValue, predictor = split
            if rid==point_reg:
                if point_reg=='0':
                    point_reg=''
                if point[predictor] <= targetValue:
                    point_reg = point_reg+str(i)+'l'
                else:
                    point_reg = point_reg+str(i)+'r'

            #print(rid, point_reg)
        #print(self.splits)
        pred=0
        if point_reg in self.region_res:
            pred = self.region_res[point_reg]
        else:
            print('Exact region missing from test, generalizing')
            while point_reg not in self.region_res and point_reg!='':
                point_reg = point_reg[:-2]
            if point_reg=='':
                if self.type == 1:
                    pred=random.randrange(1, 13, 1)
                else:
                    for reg in self.region_res:
                        pred=self.region_res[reg]
            else:
                pred = self.region_res[rid]
        return pred
    
    def printData(self):
        print('Model Splits:', self.splits)
        print('Number of regions:', len(self.regions))
        print('Number of iterations', self.train_itrs)
        print('Number of samples', self.no_samples)
        print('TRAIN DATA REGIONS')
        for reg in self.regions:
            print(reg)
    

class BaggedTrees:
    def __init__(self, train, num, typ):
        self.data = train
        self.type = typ
        self.sample_size = train.shape[0]
        self.samples = []
        self.trees = []
        for i in range(num):
            self.samples.append(self.data.sample(frac=1,replace=True))
            self.trees.append(Tree(typ=typ))
        
    def train(self, rf=False):
        for i in range(len(self.trees)):
            if rf==True:
                self.trees[i].train(self.samples[i], rf=True)
            else:
                self.trees[i].train(self.samples[i])
            self.trees[i].printData()
    
    def predict(self, test):
        size = len(self.trees)
        acc = 0
        for i in range(len(test)):
            point = test.iloc[i]
            res = []
            for j in range(size):
                res.append(self.trees[j].predictPoint(point))
            if self.type == 1:
                counts = [0]*12
                maxcount = 0
                for r in res:
                    counts[r-1]+=1
                    if counts[r-1]>counts[maxcount]:
                        maxcount=r-1
                if maxcount+1==point['month']:
                    acc+=1
            elif self.type==2:
                mean = sum(res)/len(res)
                acc+=mean
        acc/=test.shape[0]
        print('Bagging result:', acc*100)                        

def preprocessing(df):
    df = df.drop(['No'], axis = 1)
    #df = df.drop(['cbwd'], axis = 1) 
    df = df.rename(columns={'pm2.5':'pm25'})
    #df = df.dropna()
    df['pm25'] = df['pm25'].fillna((df['pm25'].median()))
    df['cbwd'] = df.cbwd.str.replace('NW', '1')
    df['cbwd'] = df.cbwd.str.replace('cv', '2')
    df['cbwd'] = df.cbwd.str.replace('NE', '3') 
    df['cbwd'] = df.cbwd.str.replace('SE', '4')
    #data['pm25'] = data['pm25'].fillna((data['pm25'].median()))
    return df

path = 'PRSA_data_2010.1.1-2014.12.31.csv'
df = pd.read_csv(path)
df = preprocessing(df)


train_years = [2012, 2014]
test_years = [2011, 2013]
traindf = df.loc[df['year'].isin(train_years)]
testdf = df.loc[df['year'].isin(test_years)]

start_time = time.time()

# Classification Decision Tree
tree = Tree(typ=1)
tree.train(traindf)
tree.printData()
tree.predict(testdf)

# Regression Decision Tree
# =============================================================================
# tree = Tree(typ=2)
# tree.train(traindf)
# tree.printData()
# tree.predict(testdf)
# 
# =============================================================================

# =============================================================================
# # Bagging - Classification Decision Tree
# num_trees= 5
# tree_type = 1   # 1 for classification, 2 for regression
# bag = BaggedTrees(traindf, num_trees, tree_type)
# bag.train()
# bag.predict(testdf)
# =============================================================================

# Bagging - Regression Decision Tree
# =============================================================================
# num_trees= 5
# tree_type = 1   # 1 for classification, 2 for regression
# bag = BaggedTrees(traindf, num_trees, tree_type)
# bag.train()
# bag.predict(testdf)
# =============================================================================

# Bagging - Classification Decision Tree with Random Forest
# =============================================================================
# num_trees= 5
# tree_type = 1   # 1 for classification, 2 for regression
# bag = BaggedTrees(traindf, num_trees, tree_type)
# bag.train(rf=True)
# bag.predict(testdf)
# 
# 
# =============================================================================
print('Total time:', time.time()-start_time)