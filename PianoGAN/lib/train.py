# train.py
# Contains training functions
# David Leung
# Wednesday 25th September - Week 9


# Import libraries
from lib.functional import to_device, ones_target, zeros_target, generator_target, device


# Train discriminator function
def train_discriminator(discriminator, loss, optimizer, realdata, fakedata):
    size = realdata.size(0)
    optimizer.zero_grad()   # reset gradients
    '''Discriminator Loss: (1/m * sum for i=1 to i=m) (log D(x(i)) + log (1 - D(G(z(i)))))'''


    '''1. Train on real data (1st half of above loss equation)'''
    # D(x(i))
    pred_real = discriminator(realdata)
    err_real = loss(pred_real, to_device(ones_target(size), device))   # real data has 1 target
    err_real.backward()

    '''2. Train on fake data (2nd half of above loss equation)'''
    # D(G(z))
    pred_fake = discriminator(fakedata)
    err_fake = loss(pred_fake, to_device(zeros_target(size), device))  # fake data has 0 target
    err_fake.backward()

    '''3. Update weights with gradients'''
    optimizer.step()

    '''4. Return error and predictions for real and fake inputs'''
    return err_real + err_fake, pred_real, pred_fake


# Train generator function
def train_generator(discriminator, loss, optimizer, fakedata):
    size = fakedata.size(0)
    optimizer.zero_grad()   # reset gradients
    '''Generator Loss: (1/m * sum for i=1 to i=m) (log (1 - D(G(z(i)))))'''


    '''1. Train on fake data'''
    # D(G(z))
    prediction = discriminator(fakedata)  # this fake data comes from generator outside of function

    '''2. Calculate error and backpropagate'''
    error = loss(prediction, to_device(generator_target(size), device))     # instead of minimizing log(1-D(G(z))), maximise log(D(gz)) for stronger gradients in early training
    error.backward()


    '''3. Update weights with gradients'''
    optimizer.step()

    '''4. Return error'''
    return error