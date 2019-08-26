import torch
import registry

@registry.register('ada','cycleGAN')
class CycleGAN:
    def __init__(self,generator,discriminator,criterions,optimizers,numTestClass):
        super(CycleGAN, self).__init__()
        self.criterion_dist = registry.construct("loss",criterions["c_dist"])
        self.criterion_GAN = registry.construct("loss",criterions["c_GAN"])
        self.criterion_cycle = registry.construct("loss",criterions["c_cyc"])
        self.criterion_identity = registry.construct("loss",criterions["c_id"])
        self.criterion_task_loss = registry.construct("loss",criterions["c_task"])
        self.G_AB = registry.construct("generator",generator)
        self.G_BA = registry.construct("generator",generator)
        self.D_A = registry.construct("disc_cls",discriminator,numTestClass)
        self.D_B = registry.construct("disc_cls",discriminator,numTestClass)
        self.optimizer_G = registry.construct("optimizer",optimizers["generators"],
                                              itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                              learning_rate)
        self.optimizer_D_A = registry.construct("optimizer",optimizers["disc_A"],
                                                D_A.parameters(),learning_rate)
        self.optimizer_D_B = registry.construct("optimizer",optimizers["disc_B"],
                                                D_B.parameters(),learning_rate)
        
    def toCuda(self):
        self.G_AB.cuda()
        self.G_BA.cuda()
        self.D_A.cuda()
        self.D_B.cuda()
        self.D_B_side.cuda()
        self.criterion_GAN.cuda()
        self.criterion_cycle.cuda()
        self.criterion_identity.cuda()
        self.criterion_task_loss.cuda()
        self.criterion_dist.cuda() 

    def load(self,gab_file,gba_file,da_file,db_file):
        self.G_AB.load_state_dict(torch.load(gab_file))
        self.G_BA.load_state_dict(torch.load(gba_file))
        self.D_A.load_state_dict(torch.load(da_file))
        self.D_B.load_state_dict(torch.load(db_file))
        
    def predict(self):
        self.D_A.eval()
        _, predictions = self.D_A(x)
        labels = torch.argmax(predictions,dim=1) # index of max along each row
        mask = torch.max(predictions,1)[0]>0.1
        return labels.detach(),mask.unsqueeze(1)

    def test_cycle(real_B,labels,test_loader):
        self.D_A.eval()
        test_loss = 0
        correct = 0
        acc_class = {}
        count_class = {}
        dt = test_loader.dataset
        samples = self.G_BA(real_B).detach()
        with torch.no_grad():
            for x in test_loader:
                x_feat, TrueLabel = x['feature'].cuda(), x['class_label'].cuda()
                _, predictions = self.D_A(x_feat)
                PredLabel= torch.argmax(predictions,dim=1).cpu() # index of max along each row
                TrueLabel= TrueLabel.cpu().numpy() 
                test_loss += np.sum( dt.testClassLabels[PredLabel].reshape(x_feat.shape[0]) - TrueLabel == 0 ) # sum up batch loss
                BatchAcc=  ( dt.testClassLabels[PredLabel].reshape(x_feat.shape[0]) - TrueLabel == 0 ) + 0           

                for i in range( 0, BatchAcc.shape[0]  ):
                    if TrueLabel[i] not in acc_class.keys():
                        acc_class[ TrueLabel[i] ]= BatchAcc[i]
                        count_class[ TrueLabel[i] ]= 1
                    else:
                        acc_class[ TrueLabel[i] ]+= BatchAcc[i]
                        count_class[ TrueLabel[i] ]+=1               

            for key in acc_class.keys():
                test_loss+= acc_class[key] / count_class[key]
            _,predictions = D_A(samples)
            PredLabel= torch.argmax(predictions,dim=1) # index of max along each row
            sample_acc = torch.sum( PredLabel - labels == 0 ).item() # sum up batch loss

    #     unseen_acc /= len(dt.Tes)
        test_loss /= len(test_loader.dataset.TestData)
        sample_acc /= labels.shape[0]


        print('\n Test set discriminator prediction: Average acc: {:.8f},\n'.format(
            test_loss))
        g_loss.append(test_loss)
        print('\n Sample set acc set: Average acc: {:.8f},\n'.format(
            sample_acc))

    def test_cycle(awa,real_B,labels,test_loader)
        awa.eval()
        self.G_BA.eval()
        test_loss = 0
        correct = 0
        acc_class = {}
        count_class = {}
        dt = test_loader.dataset
        samples = self.G_BA(real_B).detach()
        with torch.no_grad():
            C_all  = torch.Tensor(dt.AttributeData[dt.testClassLabels-1])[:,0,:].cuda()
            means,Covs = awa(C_all)
            up_means = torch.zeros(means.shape[0],means.shape[1]).cuda()
            for i in range(means.shape[0]):
                noise = torch.randn(100,means.shape[1]).cuda()
                noise_Vec = noise * Covs[i] + means[i]
                map_cluster = self.G_BA(noise_Vec)
                up_means[i,:] = torch.sum(map_cluster,0)/100
            for x in test_loader:
                x_feat, TrueLabel = x['feature'].cuda(), x['class_label'].cuda()
                PredMat= torch.zeros( (x_feat.shape[0],len(dt.testClassLabels)), dtype=torch.float32 ).cuda()
                for Iter in range(len(dt.testClassLabels)):
                    Mean= up_means[Iter,:]
                    CovarianceI= Covs[Iter,:] 

                    logDet= torch.sum(torch.log(CovarianceI))
                    logExp= -1 * torch.sum((x_feat-Mean)*CovarianceI*(x_feat-Mean),dim=1)
                    Likelihood=  logExp + logDet/2
                    PredMat[:,Iter]= Likelihood # Likelihood computed for whole batch for Iter class

                PredLabel= torch.argmax(PredMat,dim=1) # index of max along each row

                TrueLabel= TrueLabel.cpu().numpy() 
                BatchAcc=  ( dt.testClassLabels[PredLabel.cpu()].reshape(x_feat.shape[0]) - TrueLabel == 0 ) + 0           

                for i in range( 0, BatchAcc.shape[0]  ):
                    if TrueLabel[i] not in acc_class.keys():
                        acc_class[ TrueLabel[i] ]= BatchAcc[i]
                        count_class[ TrueLabel[i] ]= 1
                    else:
                        acc_class[ TrueLabel[i] ]+= BatchAcc[i]
                        count_class[ TrueLabel[i] ]+=1               

        for key in acc_class.keys():
            test_loss+= acc_class[key] / count_class[key]
        test_loss /= np.unique(TrueLabel).shape[0]
        print('\nGen Mapped Test set means knn: Average Acc: {:.8f},\n'.format(
            test_loss))
        self.G_BA.train()
        p_loss.append(test_loss)


        
    def trainGenerator(real_A,real_B,labels_A_soft,labels_B):
        
        self.optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        # GAN loss
        fake_B = self.G_AB(real_A)
        fake_B_append = torch.cat((fake_B, 
                                   labels_A_soft.unsqueeze(1)
                                   .type(torch.cuda.FloatTensor)),
                                  1)

        validity_B, pred_labels_B = self.D_B(fake_B)
        loss_GAN_AB = -torch.mean(validity_B)
        fake_A = self.G_BA(real_B)
        fake_A_append = torch.cat((fake_B,
                                   labels_B.unsqueeze(1)
                                   .type(torch.cuda.FloatTensor)),
                                  1)

        validity_A, pred_labels_A = self.D_A(fake_A)   
        loss_GAN_BA = -torch.mean(validity_A)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA ) / 2
        
#         #criterion task loss
        loss_task_B = self.criterion_task_loss(pred_labels_B * validity_B,
                                          labels_B)
        loss_task_A = self.criterion_task_loss(pred_labels_A * validity_A, 
                                          labels_A_soft)
        task_loss = (loss_task_A + loss_task_B)/2
        
        # Cycle loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.G_AB(fake_A) 
        loss_cycle_B = self.criterion_cycle(recov_B, real_B )

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G =    loss_GAN + \
                    self.lambda_cyc * loss_cycle + \
                    self.lambda_id * loss_identity + self.lmda *1e-6* task_loss
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G,loss_cycle,loss_identity,loss_GAN

    def trainDiscriminators(real_A,real_B,labels_A_soft,labels_B):
        self.optimizer_D_A.zero_grad()
            
        # -----------------------
        #  Train Discriminator A
        # -----------------------

        # Real loss
        validity_A, pred_labels_A = self.D_A(real_A)
        loss_real = -torch.mean(validity_A) + \
                     lmda * 0.2 * \
                     self.criterion_task_loss(pred_labels_A,labels_A_soft) 
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A
        validity_fake_A, pred_labels_fake_A = self.D_A(fake_A_.detach())
        if epoch > 500:
            loss_fake = torch.mean(validity_fake_A) + \
                        lmda * 0.8 * \ 
                        self.criterion_task_loss(pred_labels_fake_A,labels_B)
        else:
            loss_fake = torch.mean(validity_fake_A)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        self.optimizer_D_A.step()
        for p in D_A.parameters():
            p.data.clamp_(-0.01,0.01)
            
        # -----------------------
        #  Train Discriminator B
        # -----------------------

        self.optimizer_D_B.zero_grad()

        # Real loss
        validity_B, pred_labels_B = self.D_B(real_B)
        loss_real = -torch.mean(validity_B) + \
                    lmda * 0.2 * \
                    self.criterion_task_loss(pred_labels_B,labels_B)
        fake_B_ = fake_B
        validity_fake_B, pred_labels_fake_B = self.D_B(fake_B_.detach())
        if epoch >500:
            loss_fake = torch.mean(validity_fake_B) + \ 
                        lmda * 0.8 * \
                        self.criterion_task_loss(pred_labels_fake_B,labels_A_soft)
        else:
            loss_fake = torch.mean(validity_fake_B)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        self.optimizer_D_B.step()
        for p in D_B.parameters():
            p.data.clamp_(-0.01,0.01)
        loss_D = (loss_D_A + loss_D_B) / 2
        return loss_D


