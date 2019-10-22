# MAIN: PianoGAN.py
# Instantiation of GAN for FYP
# David Leung
# Wednesday 25th September - Week 9


# Import libraries
from lib import dataset, functional, model, train, eval, decode
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from progressbar import progressbar
from time import sleep
from torch import load

# Parameters
DATASET_NAME = 'EPiano'
SEQ_LENGTH = 600
BATCH_SIZE = 100
D_LR = 0.00008
G_LR = 0.00001
NUM_EPOCHS = 100
N_TEST_SAMPLES = 2
UPDATE_FREQUENCY = 50
SAVE_FREQUENCY = 60000


# Instantiate dataset, discriminator & generator
trainset = functional.DeviceDataLoader(DataLoader(dataset.EpianoDataset(SEQ_LENGTH), BATCH_SIZE, shuffle=True), functional.device)    # wrapped for GPU
discriminator = functional.to_device(model.Discriminator(SEQ_LENGTH), functional.device)    # wrapped for GPU
generator = functional.to_device(model.Generator(SEQ_LENGTH), functional.device)    # wrapped for GPU


# Instantiate optimizer and loss
d_optimizer = Adam(discriminator.parameters(), lr=D_LR)
g_optimizer = Adam(generator.parameters(), lr=G_LR)
loss = BCELoss()
'''
Binary Cross Entropy Loss:
L = {l1,l2....lN)^T  l(i) = -w(i) [ y(i) * log(v(i)) + (1 - y) * log(1 - v(i)) ] 
mean is calculated by computing sum(L) / N
because we don't need weights, set w(i) = 1 for all i
'''

# Instantiate logger and create new save directory
logger = eval.logger_instance(D_LR, G_LR, DATASET_NAME)
path = eval.newdir()


# # Load previous saved state_dict (OPTIONAL)
# discriminator.load_state_dict(load('./saves/2019-9-26_14-57-41/D_E0_B858.pt'))
# generator.load_state_dict(load('./saves/2019-9-26_14-57-41/G_E0_B858.pt'))
# d_optimizer.load_state_dict(load('./saves/2019-9-26_14-57-41/Dopt_E0_B858.pt'))
# g_optimizer.load_state_dict(load('./saves/2019-9-26_14-57-41/Gopt_E0_B858.pt'))


# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_num, (real_batch,_) in progressbar(enumerate(trainset)):   # index is discarded

        '''1. Train Discriminator'''
        real_data = functional.samples_to_vectors(real_batch, SEQ_LENGTH)
        # generate fake data and detach (so gradient not calculated for generator)
        fake_data = generator(functional.to_device(functional.noise(real_batch.size(0)), functional.device)).detach()
        # train discriminator
        d_error, d_pred_real, d_pred_fake = train.train_discriminator(discriminator, loss, d_optimizer, real_data, fake_data)


        '''2. Train Generator'''
        # generate fake data (no detach this time because need graidents)
        fake_data_g = generator(functional.to_device(functional.noise(real_batch.size(0)), functional.device))
        # train generator
        g_error = train.train_generator(discriminator, loss, g_optimizer, fake_data_g)


        ''' Log batch error '''
        logger.log(d_error, g_error, d_pred_real, d_pred_fake, epoch, batch_num, len(trainset))


        ''' Display progress every few batches & save model checkpoint '''
        eval.tb_update(UPDATE_FREQUENCY, epoch, NUM_EPOCHS, batch_num, len(trainset), generator, logger, N_TEST_SAMPLES, d_error, g_error, d_pred_real, d_pred_fake)
        #eval.save_param(SAVE_FREQUENCY, discriminator.state_dict(), generator.state_dict(), d_optimizer.state_dict(), g_optimizer.state_dict(), epoch, batch_num, path)

        sleep(0.000001)



