from workspace_utils import active_session
import torch

def test_model(device, model, testloader):
    '''
        Test a trained model.
    '''
    
    with active_session():
        #test phase
        accuracy = 0

        with torch.no_grad():
            print('\n### Start of network test! ###')
            model.eval()

            for image, label in testloader:
                image, label = image.to(device), label.to(device)

                test_log_ps = model(image)
                test_ps = torch.exp(test_log_ps)

                top_p, top_class = test_ps.topk(1, dim=1)
                equals = top_class == label.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
            average_accuracy = accuracy / len(testloader)
            print('\tAverage Test Set Accuracy: {:.2f}%'.format(average_accuracy.item()*100))
            print('### End of network test! ###\n')