import torch


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2, operation_reduction="mean", method_mining="random"):
        super().__init__()

        self.margin = margin
        self.method_mining = method_mining
        self.operation_reduction = operation_reduction

        self.fn_mine = getattr(self, f"mine_online_{method_mining}")

    def forward(self, input, target):
        d_ap, d_an = self.fn_mine(input, target)

        output = d_ap - d_an + self.margin
        # Cleaner than torch.maximum
        output = torch.nn.functional.relu(output)
        output = torch.sum(output) if (self.operation_reduction == "sum") else torch.mean(output)

        return output

    def mine_online_random(self, input, target):
        latent_anchor = input["anchor"]
        latent_positive = input["positive"]
        latent_negative = input["negative"]

        d_ap = (latent_anchor - latent_positive).pow(2).sum(dim=-1)
        d_an = (latent_anchor - latent_negative).pow(2).sum(dim=-1)

        return d_ap, d_an

    def mine_online_semihard_negative(self, input, target):
        latent_anchor = input["anchor"]
        latent_positive = input["positive"]
        latent_negative = input["negative"]
        target_anchor = target["anchor"]
        target_negative = target["negative"]

        d_ap = (latent_anchor - latent_positive).pow(2).sum(dim=-1)

        d_an_pairwise = (latent_anchor[:, None, ...] - latent_negative[None, ...]).pow(2).sum(dim=2)

        # Dataset sampling makes sure that at least one valid tiplet exists for each anchor
        mask_valid = target_anchor[:, None] != target_negative[None, :]
        d_an_pairwise_valid = d_an_pairwise * mask_valid.float()

        mask_semihard = d_an_pairwise_valid > (d_ap[None, :] + self.margin)
        d_an_pairwise_semihard = d_an_pairwise_valid * mask_semihard.float()

        # Set these to inf to mask out zeros from before for torch.min
        d_an_pairwise_semihard[d_an_pairwise_semihard == 0] = float("inf")
        d_an_pairwise_valid[d_an_pairwise_valid == 0] = float("inf")

        d_an_semihard, _ = torch.min(d_an_pairwise_semihard, dim=1)
        d_an_hardest, _ = torch.min(d_an_pairwise_valid, dim=1)

        mask_no_valid_semihard = d_an_semihard == float("inf")
        d_an = torch.where(mask_no_valid_semihard, d_an_hardest, d_an_semihard)

        return d_ap, d_an
