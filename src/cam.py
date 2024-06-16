import numpy as np
import random
import torch
import torch.nn.functional as F
from activations_and_gradients import ActivationsAndGradients
from image import scale_cam_image

def get_grad_cam(model, input_tensor):
    target_layers = [model.features[-1]]
    activations_and_grads = ActivationsAndGradients(
                model, target_layers)
    input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
    outputs = activations_and_grads(input_tensor)
    np_outputs = outputs.cpu().detach().numpy()
    target_classes = np.argmax(np_outputs, axis=1)

    model.zero_grad()
    loss = sum([output[target_class]
                for output, target_class in zip(outputs, target_classes)])
    loss.backward(retain_graph=True)
    
    activations_list = [a.cpu().data.numpy()
                        for a in activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                  for g in activations_and_grads.gradients]

    layer_activations = activations_list[0]
    layer_grads = grads_list[0]

    weights = np.mean(layer_grads, axis=(2, 3))

    weighted_activations = weights[:, :, None, None] * layer_activations
    cam = weighted_activations.sum(axis=1)
    relu_cam = np.maximum(cam, 0)
    grad_cam = scale_cam_image(relu_cam)
    np_outputs = None
    layer_activations = None
    layer_grads = None
    del target_layers
    activations_and_grads.release()
    del activations_and_grads
    target_classes = None
    weights = None
    weighted_activations = None
    cam = None
    relu_cam = None
    return grad_cam

def shapley_value(model, input_tensor, device='cuda', grad_cam = [], threshold = 0.5, num_samples = 1000, num_batches = 10):
    device = torch.device(device)
    grad_cam_indexes = []
    for result in grad_cam:
        grad_cam_indexes.append(set(map(tuple, np.column_stack(np.where(result > threshold)))))
    output = model.features(input_tensor).squeeze()
    np_output = output.detach().numpy()
    target_class = np.argmax(model(input_tensor).detach(), axis=1)
    num_images, num_channels, width, height = output.shape
    N = width * height
    model = model.to(device)
    values = []
    for image_index in range(num_images):
        print(image_index)
        arr = [(i, j) for i in range(width) for j in range(height)]
        np.random.seed(0)
        if len(grad_cam_indexes) != 0:
            indexes = grad_cam_indexes[image_index]
            for i in indexes:
                arr.remove(i)
        else:
            indexes = []
        Sh = np.array([np.zeros((width, height)) for i in range(num_samples)])
        pis = []
        for i in range(num_samples):
            np.random.shuffle(arr)
            pis.append(arr[:])
        avg = F.avg_pool2d(output[image_index], (width, height)).squeeze() # Cai nay la tinh gap cho 960 channel -> shape (960, )
        avg = avg.detach().numpy()
        avg_tmp = np.copy(avg)
        for x, y in indexes: # Gan san nhung pixel da duoc localize bang gradcam
            avg = (avg*N - avg_tmp + np_output[image_index, :, x, y])/N
        batches_pi = []
        batches = []
        batches_sample = []
        with torch.no_grad():
            for pi_index, pi in tqdm(enumerate(pis)):
                gap = np.copy(avg)
                batches_sample = [gap]
                for x, y in pi[:-1]:
                      gap = (gap*N - avg + np_output[image_index, :, x, y])/N
                      batches_sample.append(np.copy(gap))
                batches_pi.append(pi)
                batches.append(np.stack(batches_sample))
                if len(batches) == num_batches:
                    phis = model.classifier(torch.from_numpy(np.stack(batches)).to(device))[:, :, target_class[image_index]]
                    phis = phis.cpu().detach().numpy()
                    for phi, new_pi in zip(phis, batches_pi): # Chay cong thuc f(pre(S_i)) - f(pre(S))
                        for idx, (x, y) in enumerate(new_pi[:-1]):
                            Sh[pi_index][x][y] += phi[idx+1] - phi[idx]
                            x, y = new_pi[-1]
                        Sh[pi_index][x][y] += phi[0] - phi[-1]
                    batches = []
                    batches_pi = []

                torch.cuda.empty_cache()
            Sh = np.array(Sh).sum(axis=0)
            Sh /= num_samples
            Sh = np.maximum(Sh, 0)
            for x, y in indexes:
                Sh[x][y] = grad_cam[image_index][x][y]
            Sh = scale_cam_image(Sh[None, :, :])
        values.append(Sh)
    model = model.to('cpu')
    return np.stack(values).squeeze() # Shape (images, w, h)
