# eval.py
# Tensorboard, evaluation and generating test noise
# David Leung
# Wednesday 25th September - Week 9

# Import libraries
from lib.functional import noise, to_device, device, vectors_to_samples
from lib.utils import Logger
from torch import save
from datetime import datetime
from os import mkdir

# Logger function for tensorboard (3rd party code)
def logger_instance(D_learnrate, G_learnrate, Dataset_name):
    return Logger(model_name='GAN D_LR='+str(D_learnrate)+' G_LR'+str(G_learnrate), data_name=Dataset_name)


# Create new directory with date time naming convention
def newdir():
    dt = datetime.now()
    path = './saves/'+str(dt.year)+'-'+str(dt.month)+'-'+str(dt.day)+'_'+str(dt.hour)+'-'+str(dt.minute)+'-'+str(dt.second)+'/'
    mkdir(path)

    return path


# Save model parameters (uses HEAPS OF MEMORY ~300-600MB per save)
def save_param(save_frequency, discriminator_statedict, generator_statedict, d_optimizer_statedict, g_optimizer_statedict, epoch, batch_num, path):
    if (batch_num) % save_frequency == 0:
        save(discriminator_statedict, path+'D_E' + str(epoch) + '_B' + str(batch_num) + '.pt')
        save(generator_statedict, path+'G_E' + str(epoch) + '_B' + str(batch_num) + '.pt')
        save(d_optimizer_statedict, path+'Dopt_E' + str(epoch) + '_B' + str(batch_num) + '.pt')
        save(g_optimizer_statedict, path+'Gopt_E' + str(epoch) + '_B' + str(batch_num) + '.pt')
        print('\nsaved model parameters to: '+path+'\n')


# Tensorboard update
def tb_update(update_frequency, epoch, total_epochs, batch_num, trainset_len, generator, logger, n_test_samples, d_error, g_error, d_pred_real, d_pred_fake):
    num_test_samples = n_test_samples
    test_noise = to_device(noise(num_test_samples), device)     # Generating noise samples for evaluation

    if (batch_num) % update_frequency == 0:
        generator.eval()
        test_samples = vectors_to_samples(generator(test_noise))
        test_samples = test_samples.data
        generator.train()

        logger.log_images(test_samples.cpu(), num_test_samples, epoch, batch_num, trainset_len)

        # Display status logs
        logger.display_status(epoch, total_epochs, batch_num, trainset_len, d_error, g_error, d_pred_real, d_pred_fake)
