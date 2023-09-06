from itertools import product, combinations

import torch
import torch.nn as nn


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, device, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._device = device
        self._epsilon = epsilon

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
        """
        Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """

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
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1).to(self._device)
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

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        concatenated_quantized = (
            self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()]
            if not self.training or record_codebook_stats
            else None
        )

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = commitment_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # # Convert quantized from BHWC -> BCHW
        return {
            "vq_loss": vq_loss,
            "quantized": quantized.permute(2, 0, 1).contiguous(),
            "perplexity": perplexity,
            "encodings": encodings.view(batch_size, time, -1),
            "distances": distances.view(batch_size, time, -1),
            "encoding_indices": encoding_indices,
            "losses": {"vq_loss": vq_loss.item()},
            "encoding_distances": encoding_distances,
            "embedding_distances": embedding_distances,
            "frames_vs_embedding_distances": frames_vs_embedding_distances,
            "concatenated_quantized": concatenated_quantized,
        }

    @property
    def embedding(self):
        return self._embedding
