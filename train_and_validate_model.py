from workspace_utils import active_session
import torch

def train_and_validate_model(device, model, criterion, optimizer, epochs, already_trained_epochs, trainloader, validationloader):
    '''
        Trains and validates a model.
    '''
    
    batches_results_every = 5
    
    #prevent workspace from being idle
    with active_session():
        print('\n### Start of network training! ###')
        print('\tAttention: Log results will be printed every {} steps\n'.format(batches_results_every))
        
        if already_trained_epochs > 0:
            print("\tAlready trained epochs = {}".format(already_trained_epochs))
            print("\tTo-be-trained epochs = {}\n".format(epochs))
            
        model.train()

        for epoch in range(epochs):

            #train phase
            train_loss = 0
            step = 0

            for images, labels in trainloader:
                step += 1

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                #enter validation phase every "batches_results_every" batches
                if step % batches_results_every == 0:
                    #validation phase
                    validation_loss = 0
                    accuracy = 0

                    with torch.no_grad():
                        model.eval()

                        for images, labels in validationloader:

                            images, labels = images.to(device), labels.to(device)

                            validation_log_ps = model(images)

                            validation_loss += criterion(validation_log_ps, labels)

                            validation_ps = torch.exp(validation_log_ps)

                            top_p, top_class = validation_ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)

                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    average_accuracy = accuracy / len(validationloader)

                    print('\tStep: {}; Epoch {}/{}; Train Loss: {:.2f}; Validation Loss: {:.2f}; Accuracy: {:.2f}%'
                          .format(step,
                                  epoch+1,
                                  epochs,
                                  train_loss / batches_results_every,
                                  validation_loss / len(validationloader),
                                  average_accuracy.item()*100))

                    train_loss = 0

                    model.train()

        print('### End of network training! ###')