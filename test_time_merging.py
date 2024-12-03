import torch
from torch import nn, load
from torch.nn import KLDivLoss
from torchvision import transforms
from tqdm import tqdm

from datasets.cars import PytorchStanfordCars, Cars
from src.task_vectors import TaskVector


def freeze_all_but_last_layers(model: nn.Module, n_layers: int = 1):
    for param in model.parameters():
        param.requires_grad = False

    for i in range(1, n_layers + 1):
        print(f"frozen weight {-i}")
        for pr in model.model.visual.transformer.resblocks[-i].parameters():
            pr.requires_grad = True

    return model


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


# TODO: probably not needed
# class TestTimeMerging(nn.Module):
#     def __init__(self, pretrained_model, finetuned_model, merged_model, last_layers_update: int = 1):
#         super(TestTimeMerging, self).__init__()
#         self.pretrained_model = freeze(pretrained_model)
#         self.finetuned_model = freeze(finetuned_model)
#         self.merged_model = freeze_all_but_last_layers(merged_model, n_layers=last_layers_update)
#         self.last_layers_update = last_layers_update
#
#     def forward(self, x):
#         x_pretrained = self.pretrained_model(x)
#         x_finetuned = self.finetuned_model(x)
#         x_merged = self.merged_model(x)
#         return x_pretrained, x_finetuned, x_merged


def resolve_gradients(grad_loss_p_m, grad_loss_f_m):
    if grad_loss_p_m.shape != grad_loss_f_m.shape:
        raise ValueError("Tensors must have the same shape")

    result = torch.empty_like(grad_loss_p_m)

    # Case 1: Different signs
    mask_diff_signs = (grad_loss_p_m * grad_loss_f_m) < 0
    result[mask_diff_signs] = torch.max(torch.abs(grad_loss_p_m[mask_diff_signs]), torch.abs(grad_loss_f_m[mask_diff_signs]))

    # Case 2: Same signs
    mask_same_signs = ~mask_diff_signs
    result[mask_same_signs] = (grad_loss_p_m[mask_same_signs] + grad_loss_f_m[mask_same_signs]) / 2

    return result








def train(model_p, model_m, model_f, dataloader, optimizer, epochs, n_iterations, device):
    model_m.train()
    criterion_p_m = KLDivLoss(reduction="batchmean")
    criterion_f_m = KLDivLoss(reduction="batchmean")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch+1}/{epochs}")
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            for i in range(n_iterations):
                print(f"iteration {i+1}")
                optimizer.zero_grad()

                pretrained_output = model_p(images).detach()
                merged_output = model_m(images)
                loss_p_m = criterion_p_m(merged_output, pretrained_output)
                loss_p_m.backward(retain_graph=True)
                # TODO: hardcoded last layer of Transformer
                # TODO: fix gradient operations (maybe have to work directly on .parameters())
                grads_loss_p_m = model_m.model.visual.transformer.resblocks[-1].grad.clone()

                optimizer.zero_grad()
                finetuned_output = model_f(images).detach()
                loss_f_m = criterion_f_m(merged_output, finetuned_output)
                loss_f_m.backward(retain_graph=True)
                # TODO: hardcoded last layer of Transformer
                grad_loss_f_m = model_m.model.visual.transformer.resblocks[-1].grad.clone()

                resolved_gradients = resolve_gradients(grads_loss_p_m, grad_loss_f_m)
                # optimizer.zero_grad()
                model_m.model.visual.transformer.resblocks[-1].grad = resolved_gradients

                optimizer.step()

                # running_loss += loss.item()
                #
                # _, predicted = torch.max(outputs, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()


    return model_p, model_m, model_f




if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: figure out if we can overcome specific resize dims
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((240,240))])
    cars = Cars(preprocess=transform, location="data", batch_size=512)
    test_loader = cars.test_loader

    pretrained_path = "checkpoints/ViT-B-32/zeroshot.pt"
    finetuned_path = "checkpoints/ViT-B-32/Cars/finetuned.pt"

    task_vector = TaskVector(pretrained_path, finetuned_path)
    merged = freeze_all_but_last_layers(task_vector.apply_to(pretrained_path, scaling_coef=0.5))
    # print(merged.model.visual.transformer.resblocks[-1].requires_grad)
    pretrained = freeze(torch.load(pretrained_path))
    finetuned = freeze(torch.load(finetuned_path))

    # test_time_merging_model = TestTimeMerging(torch.load(pretrained_path), torch.load(finetuned_path), merged)
    optimizer = torch.optim.Adam(merged.model.visual.transformer.resblocks[-1].parameters(), lr=1e-3)
    p, m, f = train(pretrained.to(DEVICE), merged.to(DEVICE), finetuned.to(DEVICE), test_loader, optimizer, 10, 10, DEVICE)
    torch.save(m.merged_model.state_dict(), "checkpoints/first_try.pth")




    # # for block in model.model.visual.transformer.resblocks:
    # #     print(block)
    # #     break
    #
    # print(model.model.visual.transformer.resblocks[-1])