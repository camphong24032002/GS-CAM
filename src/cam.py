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

def shapley_value(model, input_tensor, device='cuda', grad_cam = [], threshold = 0.5, num_samples = 10000, num_batches = 100):
    with torch.no_grad():
        grad_cam_indexes = []
        # Get indexes of pixels that have the gradient over the threshold
        for result in grad_cam:
            grad_cam_indexes.append(set(map(tuple, np.column_stack(np.where(result > threshold)))))
        # Get feature map
        output = model.features(input_tensor)
        np_output = output.cpu().detach().numpy()
        
        # Get target class
        origin_output = model(input_tensor).cpu().detach().numpy()
        target_class = np.argmax(origin_output, axis=1)
        origin_output = None
        num_images, num_channels, width, height = output.shape
        N = width * height
        values = []
        # Loop through each image
        for image_index in range(num_images):
            # Get pixels that have the gradient lower than the threshold to apply shapley technique
            arr = [(i, j) for i in range(width) for j in range(height)]
            np.random.seed(0)
            if len(grad_cam_indexes) != 0:
                indexes = grad_cam_indexes[image_index]
                for i in indexes:
                    arr.remove(i)
            else:
                indexes = []
            # Sh is used to store the shapley value of each pixel
            Sh = np.zeros((num_samples, width, height))
            pis = []
            # Generate random samples
            for i in range(num_samples):
                random.shuffle(arr)
                pis.append(arr[:])
            del arr
            # Calculate the average of the feature map
            avg = F.avg_pool2d(output[image_index], (width, height)).squeeze() # Cai nay la tinh gap cho 960 channel -> shape (960, )
            avg = avg.cpu().detach().numpy()
            avg_tmp = np.copy(avg)
            # Ignore selected pixel
            for x, y in indexes:
                avg = (avg*N - avg_tmp + np_output[image_index, :, x, y])/N
            avg_tmp = None
            batches_pi = []
            batches_sample = []
            batches = []
            # Loop through each sample
            for pi_idx, pi in enumerate(pis):
                gap = np.copy(avg)
                batches_sample = []
                # Calculate the gap of the feature map when adding a pixel
                for x, y in pi:
                    gap = (gap*N - avg + np_output[image_index, :, x, y])/N
                    batches_sample.append(np.copy(gap))
                gap = None
                batches_pi.append(pi)
                batches.append(np.stack(batches_sample))
                del batches_sample
                batches_sample = []
                if len(batches_pi) == num_batches:
                    np_batches = np.stack(batches)
                    tmp = torch.from_numpy(np_batches).to(device)
                    phis = model.classifier(tmp).softmax(dim=-1)[:, :, target_class[image_index]]
                    phis = phis.cpu().detach().numpy()
                    # Calculate the shapley value of each pixel
                    for pi2_idx, (pi2, phi) in enumerate(zip(batches_pi, phis)):
                        sample_idx = pi_idx - (num_batches - 1) + pi2_idx
                        for idx, (x, y) in enumerate(pi2[:-1]):
                            Sh[sample_idx][x][y] += phi[idx] - phi[idx-1]
                    np_batches = None
                    del batches
                    batches = []
                    phis = None
                    del batches_pi
                    batches_pi = []
            mean_Sh = Sh.mean(axis=0)
            relu_Sh = np.maximum(mean_Sh, 0)
            relu_Sh = relu_Sh[np.newaxis, :, :]
            scaled_Sh = scale_cam_image(relu_Sh)
            for x, y in indexes:
                scaled_Sh[:, x, y] = grad_cam[image_index][x][y]
            values.append(scaled_Sh)
            del indexes
            avg = None
            Sh = None
            mean_Sh = None
            relu_Sh = None
            scaled_Sh = None
            del pis
        np_values = np.stack(values).squeeze(1)
        del values
        output = None
        np_output = None
        del grad_cam_indexes
        return np_values # Shape (images, w, h)