import torch
from torch import nn
from torch.nn import KLDivLoss
from torchvision import transforms
from tqdm import tqdm

import datasets.cars
from datasets.cars import Cars
from src.task_vectors import TaskVector
import wandb

class CombinedModel(nn.Module):
    def __init__(self, base_model, classification_head):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.classification_head = classification_head

    def forward(self, x):
        features = self.base_model(x)  # Extract features from the base model
        logits = self.classification_head(features)  # Pass through the classification head
        return logits


def freeze_all_but_last_layers(model: nn.Module, n_layers: int = 1):
    for param in model.parameters():
        param.requires_grad = False

    for i in range(1, n_layers + 1):
        for pr in model.base_model.model.visual.transformer.resblocks[-i].parameters():
            pr.requires_grad = True

    return model


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    return model


def resolve_gradients(list_grad_loss_p_m, list_grad_loss_f_m):
    if len(list_grad_loss_p_m) != len(list_grad_loss_f_m):
        raise ValueError("Lists of tensors must have the same length")

    result_param_list = []
    for t in range(11, len(list_grad_loss_p_m)):
        grad_loss_p_m = list_grad_loss_p_m[t]
        grad_loss_f_m = list_grad_loss_f_m[t]
        if grad_loss_p_m.shape != grad_loss_f_m.shape:
            raise ValueError("Tensors must have the same shape")

        # Case 1: Different signs
        mask_diff_signs = (grad_loss_p_m * grad_loss_f_m) < 0
        maxes = torch.max(torch.abs(grad_loss_p_m), torch.abs(grad_loss_f_m))

        p_m_signs = (maxes == torch.abs(grad_loss_p_m)) * torch.sign(grad_loss_p_m)
        f_m_signs = (maxes == torch.abs(grad_loss_f_m)) * torch.sign(grad_loss_f_m)
        final_signs = p_m_signs.int() | f_m_signs.int()

        result = maxes * final_signs

        # Case 2: Same signs
        mask_same_signs = ~mask_diff_signs
        result[mask_same_signs] = (grad_loss_p_m[mask_same_signs] + grad_loss_f_m[mask_same_signs]) / 2

        result_param_list.append(result)

    return result_param_list


def log_accuracy(iteration, labels, merged_output, finetuned_output, pretrained_output):
    _, predicted_merged = torch.max(merged_output, 1)
    correct_preds_merged = (predicted_merged == labels).sum().item()
    print(f"[MERGED] Correct predictions in iteration {iteration + 1}: {correct_preds_merged}/{labels.size(0)}")
    wandb.log({"iteration": iteration, "mer_correct_prediction_per": correct_preds_merged/labels.size(0)*100, "lr": optimizer.param_groups[0]["lr"]})

    # these are frozen so just plot them once
    if iteration == 0:
        _, predicted_finetuned = torch.max(finetuned_output, 1)
        correct_preds_finetuned = (predicted_finetuned == labels).sum().item()
        _, predicted_pretrained = torch.max(pretrained_output, 1)
        correct_preds_pretrained = (predicted_pretrained == labels).sum().item()
        print(
            f"[FINETUNED] Correct predictions in iteration {iteration + 1}: {correct_preds_finetuned}/{labels.size(0)}")
        print(
            f"[PRETRAINED] Correct predictions in iteration {iteration + 1}: {correct_preds_pretrained}/{labels.size(0)}")
        wandb.log({"fin_correct_prediction_per":correct_preds_finetuned/ labels.size(0)*100, "pre_correct_prediction_per":correct_preds_pretrained/labels.size(0)*100 })


def train(model_p, model_m, model_f, dataloader, optimizer, epochs, n_iterations, device):
    model_m.train()
    criterion_p_m = KLDivLoss(reduction="batchmean")
    criterion_f_m = KLDivLoss(reduction="batchmean")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            for i in range(n_iterations):
                optimizer.zero_grad()

                pretrained_output = model_p(images).detach()
                merged_output = model_m(images)

                loss_p_m = criterion_p_m(merged_output, pretrained_output)
                loss_p_m.backward(retain_graph=True)
                # TODO: hardcoded last layer of Transformer
                merged_model_last_resblock = model_m.base_model.model.visual.transformer.resblocks[-1]
                # print(list(merged_model_last_resblock.parameters()))
                p_m_grads = []
                for param in merged_model_last_resblock.parameters():
                    p_m_grads.append(param.grad)

                optimizer.zero_grad()
                finetuned_output = model_f(images).detach()
                loss_f_m = criterion_f_m(merged_output, finetuned_output)
                loss_f_m.backward(retain_graph=True)
                # TODO: hardcoded last layer of Transformer
                f_m_grads = []
                for param in merged_model_last_resblock.parameters():
                    f_m_grads.append(param.grad)

                optimizer.zero_grad()
                resolved_gradients = resolve_gradients(p_m_grads, f_m_grads)
                print(len(resolved_gradients[0]))
                print("---------")
                print(len(list(merged_model_last_resblock.parameters())[11]))
                # for idx, param in enumerate(list(merged_model_last_resblock.parameters())[11]):
                #     print(idx)
                #     param.grad = resolved_gradients[idx]
                list(merged_model_last_resblock.parameters())[11].grad = resolved_gradients[0]
                optimizer.step()

                log_accuracy(i, labels, merged_output, finetuned_output, pretrained_output)


            _, predicted = torch.max(merged_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Accuracy in epoch {epoch + 1}: {accuracy:.2f}%")
        wandb.log({"epoch": epoch, "epoch_acc": accuracy})

    return model_p, model_m, model_f


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "device": DEVICE,
        "resize": (240, 240),
        "dataset": "cars",
        "batch_size": 512,
        "arch": "ViT-B-32",
        "scaling_coef": 0.5,
        "lr": 1e-3,
        "epochs": 10,
        "n_iterations": 10
    }
    # TODO: figure out if we can overcome specific resize dims
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(config["resize"])])
    cars = Cars(preprocess=transform, location="data", batch_size=config["batch_size"])
    test_loader = cars.test_loader

    pretrained_path = "checkpoints/"+config["arch"]+"/zeroshot.pt"
    finetuned_path = "checkpoints/"+config["arch"]+"/Cars/finetuned.pt"
    cars_head_path = "checkpoints/"+config["arch"]+"/head_Cars.pt"

    pretrained_backbone = torch.load(pretrained_path)
    finetuned_backbone = torch.load(finetuned_path)

    wandb.init(entity="illia-dovhalenko-jagiellonian-university", project="pattern", config=config,
               name=f"Model merging 1 try, arch={config['arch']}, epochs={config['epochs']}, scaling_coef={config['scaling_coef']}")

    task_vector = TaskVector(pretrained_path, finetuned_path)
    merged_backbone = task_vector.apply_to(pretrained_path, scaling_coef=config["scaling_coef"])

    cars_classification_head = torch.load(cars_head_path)
    pretrained = freeze(CombinedModel(pretrained_backbone, cars_classification_head))
    finetuned = freeze(CombinedModel(finetuned_backbone, cars_classification_head))
    merged = freeze_all_but_last_layers(CombinedModel(merged_backbone, cars_classification_head))

    optimizer = torch.optim.Adam(merged.base_model.model.visual.transformer.resblocks[-1].parameters(), lr=config["lr"])
    p, m, f = train(pretrained.to(DEVICE), merged.to(DEVICE), finetuned.to(DEVICE), test_loader, optimizer, 10, 10,
                    DEVICE)
    torch.save(m.merged_model.state_dict(), "checkpoints/first_try.pth")
