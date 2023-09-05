from itertools import product, combinations

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, device):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost
        self._device = device

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(1, 2, 0).contiguous()
        input_shape = inputs.shape
        _, time, batch_size = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Compute distances between encoded audio frames and embedding vectors
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)

        # Compute distances between encoding vectors
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(flat_input, r=2)]
            encoding_distances = torch.tensor(_encoding_distances).to(self._device).view(batch_size, -1)
        else:
            encoding_distances = None

        # Compute distances between embedding vectors
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [
                torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(self._embedding.weight, r=2)
            ]
            embedding_distances = torch.tensor(_embedding_distances).to(self._device)
        else:
            embedding_distances = None

        # Sample nearest embedding
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [
                torch.dist(items[0], items[1], 2).to(self._device)
                for items in product(flat_input, self._embedding.weight.detach())
            ]
            frames_vs_embedding_distances = (
                torch.tensor(_frames_vs_embedding_distances).to(self._device).view(batch_size, time, -1)
            )
        else:
            frames_vs_embedding_distances = None

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        concatenated_quantized = (
            self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()]
            if not self.training or record_codebook_stats
            else None
        )

        # Losses
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = q_latent_loss + commitment_loss

        quantized = inputs + (quantized - inputs).detach()  # Trick to prevent backpropagation of quantized
        avg_probs = torch.mean(encodings, dim=0)

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  # Exponential entropy

        # Convert quantized from BHWC -> BCHW

        return {
            "vq_loss": vq_loss,
            "quantized": quantized.permute(2, 0, 1).contiguous(),
            "perplexity": perplexity,
            "encodings": encodings.view(batch_size, time, -1),
            "distances": distances.view(batch_size, time, -1),
            "encoding_indices": encoding_indices,
            "losses": {
                "e_latent_loss": e_latent_loss.item(),
                "q_latent_loss": q_latent_loss.item(),
                "commitment_loss": commitment_loss.item(),
                "vq_loss": vq_loss.item(),
            },
            "encoding_distances": encoding_distances,
            "embedding_distances": embedding_distances,
            "frames_vs_embedding_distances": frames_vs_embedding_distances,
            "concatenated_quantized": concatenated_quantized,
        }

    @property
    def embedding(self):
        return self._embedding
