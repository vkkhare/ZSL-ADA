import torch
import registry

class BaseZSL:
    def __init__(self, base_model,optimizer,decay,device):
        self.model = registry.construct('base_model',base_model).to(device)
        self.optimizer = registry.construct('optimizer',optimizer,[
        {'params': filter(lambda p: p.requires_grad, self.model.FC_layer_m.parameters()),'weight_decay': decay['fc_m']},
        {'params': filter(lambda p: p.requires_grad, self.model.FC_layer_c.parameters()),'weight_decay': decay['fc_c']}
    ],lr=learning_rate)
        self.device = device
        
    def load(self,file):
        if os.path.isfile(file):
            print("=> loading checkpoint '{}'".format(file))
            checkpoint = torch.load(file)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(file))
        else:
            print("=> no checkpoint found at '{}'".format(file))
            return 0
        
    def train_zsl(self, train_loader, epoch,losslist):
        self.model.train()
        b_idx = 0
        for x in train_loader:
            b_idx+=1
            x_feat = x['feature'].to(self.device)
            label = x['class_label'].type(torch.LongTensor).to(self.device)
            attribute = x['attribute'].to(self.device)
            self.optimizer.zero_grad()
            means,covs = self.model(attribute)
            loss_eval = torch.sum((x_feat-means)*covs*(x_feat-means)) - torch.sum(torch.log(covs))/2
            loss_eval.backward()
            self.optimizer.step()

            if b_idx%6 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, b_idx * x_feat.shape[0], len(train_loader.dataset),  100. * b_idx / len(train_loader), loss_eval.item()))

        losslist.append(loss_eval.item())
        
    def predict_labels(self,x_feat,dataset):
        self.model.eval()
        with torch.no_grad():
            C_all  = torch.Tensor(dataset.AttributeData[dataset.testClassLabels-1])[:,0,:].cuda()
            means,Covs = self.model(C_all)
            PredMat= torch.zeros( (x_feat.shape[0],len(dataset.testClassLabels)), dtype=torch.float32 ).cuda()
            class_probab = torch.ones(len(dataset.testClassLabels)).cuda() / len(dataset.testClassLabels)
            for Iter in range(len(dataset.testClassLabels)):
                Mean= means[Iter,:]
                CovarianceI= Covs[Iter,:] 

                logDet= torch.sum(torch.log(CovarianceI))
                logExp= -1 * torch.sum((x_feat-Mean)*CovarianceI*(x_feat-Mean),dim=1)
                Likelihood=  logExp + logDet/2
                PredMat[:,Iter]= Likelihood # Likelihood computed for whole batch for Iter class

            P_c_x = F.softmax(PredMat,1)
            mask = torch.max(P_c_x,1)[0] > 0.5
            labels = torch.argmax(P_c_x,dim=1)
        return labels.detach(),mask.unsqueeze(1)

    def test(self, test_loader,epoch,losslist, acc_class, count_class):
        self.model.eval()       
        test_loss = 0
        test_loss_2 = 0
        correct = 0
        with torch.no_grad():
            dt = test_loader.dataset
            C_all  = torch.Tensor(dt.AttributeData[dt.testClassLabels-1])[:,0,:].to(device)
            means,Covs = self.model(C_all)

            for x in test_loader:
                x_feat, TrueLabel = x['feature'].to(device), x['class_label'].to(device)
                PredMat= torch.zeros( (x_feat.shape[0],len(dt.testClassLabels)), dtype=torch.float32 ).to(device)
                for Iter in range(len(dt.testClassLabels)):
                    Mean= means[Iter,:]
                    CovarianceI= Covs[Iter,:] 

                    logDet= torch.sum(torch.log(CovarianceI))
                    logExp= -1 * torch.sum((x_feat-Mean)*CovarianceI*(x_feat-Mean),dim=1)
                    Likelihood=  logExp + logDet/2
                    PredMat[:,Iter]= Likelihood # Likelihood computed for whole batch for Iter class

                PredLabel= torch.argmax(PredMat,dim=1) # index of max along each row
                PredLabel= PredLabel.cpu().numpy()
                TrueLabel= TrueLabel.cpu().numpy() 
                BatchAcc=  ( dt.testClassLabels[PredLabel].reshape(x_feat.shape[0]) - TrueLabel == 0 ) + 0           

                for i in range( 0, BatchAcc.shape[0]  ):
                    if TrueLabel[i] not in acc_class.keys():
                        acc_class[ TrueLabel[i] ]= BatchAcc[i]
                        count_class[ TrueLabel[i] ]= 1
                    else:
                        acc_class[ TrueLabel[i] ]+= BatchAcc[i]
                        count_class[ TrueLabel[i] ]+=1               

                test_loss_2 += np.sum( BatchAcc )
                #test_loss += torch.sum( dt.testClassLabels[PredLabel].reshape(x_feat.shape[0]) - TrueLabel != 0 ).item() # sum up batch loss

            print(dt.testClassLabels[PredLabel].reshape(x_feat.shape[0]))
            print("means \n",means,"means>0",means[means>1].shape,"covs \n",Covs)

        for key in acc_class.keys():
            test_loss+= acc_class[key] / count_class[key]

        print(acc_class, '\n')   
        test_loss /= len( dt.testClassLabels )
        test_loss_2 /= len(test_loader.dataset.TestData)
        print('\n Test set: Average loss: {:.8f},\n'.format(
            test_loss))

        losslist.append(test_loss)

