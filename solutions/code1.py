epsilons=[i/10 for i in range(20)]

adv_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False)

examples = enumerate(adv_loader)
batch_idx, (images, labels) = next(examples)
images=images.cuda()
labels=labels.cuda()

models=[CNN,SNN1,SNN2]
CNN_acc=[]
SNN1_acc=[]
SNN2_acc=[]

for i,model in enumerate(models):
  fmodel = fb.PyTorchModel(model, bounds=(0, 1))
  attack = fb.attacks.PGD()
  for eps in epsilons:
    _, advs, success = attack(fmodel, images, labels, epsilons=eps)
    robust_accuracy = 1 - success.float().mean(axis=-1)
    if i==0:
      CNN_acc.append(robust_accuracy.item())
    elif i==1:
      SNN1_acc.append(robust_accuracy.item())
    else:
      SNN2_acc.append(robust_accuracy.item())


plt.figure()
plt.plot(epsilons,CNN_acc,marker='o',label='CNN')
plt.plot(epsilons,SNN1_acc,marker='o',label='SNN, Uth=1, seql=64')
plt.plot(epsilons,SNN2_acc,marker='o',label='SNN, Uth=1.75, seql=48')
plt.ylabel('Accuracy')
plt.xlabel('Noise budget (epsilon)')
plt.legend()